"""Full-data, predeclared leakage comparison; no production/model edits."""
import csv
import datetime
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

base=Path('/mnt/e/Company/_codex_research_20261106')
evidence=base/'eda'
repo=base/'work/eda/openram-leakage'
protocol=evidence/'leakage-evaluation-protocol.json'
assert protocol.exists()
os.environ['OPENRAM_HOME']=str(repo/'compiler')
os.environ['OPENRAM_TECH']=str(repo/'technology')
spec=importlib.util.spec_from_file_location('openram',repo/'__init__.py',submodule_search_locations=[str(repo)])
openram=importlib.util.module_from_spec(spec)
sys.modules['openram']=openram
spec.loader.exec_module(openram)
openram.init_openram(str(evidence/'numeric_config.py'),is_unit_test=True)
from openram.characterizer import linear_regression
regression=importlib.import_module('openram.characterizer.regression_model')
from openram.characterizer.analytical_util import get_data, get_data_names, get_scaled_data, unscale_data
import numpy as np
import scipy
import sklearn
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

def metrics(y,p):
    rel=np.abs(p-y)/y
    return {'r2_pooled':float(r2_score(y,p)), 'mae':float(mean_absolute_error(y,p)),
            'median_relative_abs_error':float(np.median(rel)),
            'p95_relative_abs_error':float(np.percentile(rel,95)),
            'max_relative_abs_error':float(np.max(rel)),
            'negative_predictions':int(np.sum(p<0)), 'nonpositive_predictions':int(np.sum(p<=0)),
            'prediction_min':float(np.min(p)), 'prediction_max':float(np.max(p))}

def controlled(log=False):
    estimator=make_pipeline(MinMaxScaler(),Ridge(alpha=1.0))
    return TransformedTargetRegressor(regressor=estimator,func=np.log,inverse_func=np.exp) if log else estimator

def evaluate(tech,protocol_name,arm,x,y,splits,model,global_scaled=False,ys=None):
    predictions=np.full(y.shape,np.nan)
    folds=[]
    lo,hi=float(y.min()),float(y.max())
    for fold,(train,test) in enumerate(splits):
        fit=clone(model)
        fit.fit(x[train],ys[train] if global_scaled else y[train])
        p=fit.predict(x[test])
        if global_scaled:p=p*(hi-lo)+lo
        assert np.isfinite(p).all()
        predictions[test]=p
        fm=metrics(y[test],p)
        fm.update(fold=fold,train_count=len(train),test_count=len(test),test_indices=test.tolist(),target_variance=float(np.var(y[test])))
        folds.append(fm)
    assert np.isfinite(predictions).all()
    result=metrics(y,predictions)
    result.update(mean_fold_r2=float(np.mean([f['r2_pooled'] for f in folds])),folds=folds)
    p=evidence/f'{tech}-{protocol_name}-{arm}-oof.csv'
    fold_by_row={int(i):f for f,(_,test) in enumerate(splits) for i in test}
    with p.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['row_index','fold','true_leakage_csv_units','predicted_leakage_csv_units'])
        w.writerows((i,fold_by_row[i],float(y[i]),float(predictions[i])) for i in range(len(y)))
    result['oof_file']=p.name
    return result

report={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'frozen_protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
        'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
        'technologies':[]}
for tech in ['freepdk45','scn4m_subm']:
    path=repo/'technology'/tech/'sim_data/sim_data.csv'
    regression.data_path=str(path)
    names=get_data_names(str(path))
    raw=np.asarray(get_data(str(path)),dtype=float).T
    scaled=get_scaled_data(str(path))
    assert names[:4]==['num_words','word_size','words_per_row','local_array_size'],names[:4]
    col=names.index('leakage_power');x,y=raw[:,:9],raw[:,col]
    assert np.isfinite(raw).all() and np.all(y>0)
    groups,group_ids=np.unique(x[:,:4],axis=0,return_inverse=True)
    splits=list(KFold(n_splits=10,shuffle=False).split(x))
    grouped_splits=list(LeaveOneGroupOut().split(x,y,group_ids))
    m=linear_regression.__new__(linear_regression);m.num_inputs=9
    models=m.train_models()
    actual_fit=np.asarray(unscale_data(models['leakage_power'].predict(scaled[:,:9]).tolist(),str(path),pos=col))
    item={'technology':tech,'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'rows':len(y),'feature_names':names[:9],'geometry_groups':groups.tolist(),
          'actual_production_full_fit':metrics(y,actual_fit),'protocols':{}}
    arms=[('global_reference',m.get_model(),True),('controlled_ridge',controlled(),False),('log_target_ridge',controlled(True),False)]
    for protocol_name,folds in [('kfold10',splits),('leave_geometry_out',grouped_splits)]:
        item['protocols'][protocol_name]={}
        for arm,model,global_scaled in arms:
            if global_scaled and protocol_name!='kfold10':continue
            result=evaluate(tech,protocol_name,arm,scaled[:,:9] if global_scaled else x,y,folds,model,global_scaled,scaled[:,col])
            item['protocols'][protocol_name][arm]=result
            print(json.dumps({'tech':tech,'protocol':protocol_name,'arm':arm,**{k:v for k,v in result.items() if k not in ['folds']}}),flush=True)
    fitted=controlled().fit(x,y).predict(x)
    item['controlled_vs_actual_full_fit_max_abs_diff']=float(np.max(np.abs(fitted-actual_fit)))
    assert np.allclose(fitted,actual_fit,rtol=1e-12,atol=1e-12)
    candidate=controlled(True).fit(x,y).predict(x)
    item['candidate_full_fit']=metrics(y,candidate)
    k=item['protocols']['kfold10'];g=item['protocols']['leave_geometry_out']
    item['frozen_gate']={
        'kfold_mean_r2_improved':k['log_target_ridge']['mean_fold_r2']>k['controlled_ridge']['mean_fold_r2'],
        'grouped_pooled_r2_improved':g['log_target_ridge']['r2_pooled']>g['controlled_ridge']['r2_pooled'],
        'kfold_mae_not_worse':k['log_target_ridge']['mae']<=k['controlled_ridge']['mae'],
        'grouped_mae_not_worse':g['log_target_ridge']['mae']<=g['controlled_ridge']['mae'],
        'no_nonpositive_oof_or_fit':all(z['log_target_ridge']['nonpositive_predictions']==0 for z in [k,g]) and item['candidate_full_fit']['nonpositive_predictions']==0}
    report['technologies'].append(item)
    (evidence/'leakage-comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
report['passes_all_frozen_numerical_gates']=all(all(t['frozen_gate'].values()) for t in report['technologies'])
report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(evidence/'leakage-comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
print('FROZEN_GATES',report['passes_all_frozen_numerical_gates'],flush=True)
openram.end_openram()
