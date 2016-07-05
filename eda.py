import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/Seattle_code_violations_database.csv')

df.columns = [ col.replace(' ', '_').lower() for col in df.columns ]
'''
u'case_number', u'case_type', u'address', u'description', u'case_group',
      u'date_case_created', u'last_inspection_date',
      u'last_inspection_result', u'status']
'''

case_type = set(df.case_type.tolist())
'''
 {'CITATION',
 'CONDO CONVERSION AND PRESALE',
 'CONSTRUCTION',
 'HOUSING / ZONING',
 'NOISE/SOUND LEVELS',
 'TENANT RELOCATION ORDINANCE',
 'UNFIT BUILDING OR PREMISES',
 'VACANT BUILDING MONITORING'}
'''
train_df = pd.get_dummies(df.case_type)

case_group = set(df.case_group.tolist())
'''
{nan,
 'BUILDING',
 'BUILDING AND PREMISES',
 'CONDO/COOP CONVERSION',
 'CONSTRUCTION NOISE',
 'ELECTRICAL',
 'HOUSING',
 'JUST CAUSE EVICTION',
 'MECHANICAL',
 'NONCONSTRUCTION NOISE',
 'OTHER CONSTRUCTION',
 'PREMISES',
 'PRESALE',
 'SIGNS',
 'SITE',
 'TENANT RELOCATION ASSIST ORD',
 'TRAO AVOIDANCE',
 'VACANT BUILDING',
 'WEEDS AND VEGETATION',
 'ZONING'}
'''

last_inspection_result = set(df.last_inspection_result.tolist())
'''
{nan, 'FAILED', 'HOLD', 'PASSED'}
'''

status = sorted(set(df.status.tolist()))
'''
{'ADMINISTRATIVE CLOSURE',
 'CLOSED',
 'COMPLAINT/APPLICATN WITHDRAWN',
 'ENFORCED COMPLIANCE',
 'NO VIOLATION',
 'OPEN',
 'TRANSFERRED TO EXTERNAL AGENCY',
 'VIOLATION',
 'VOLUNTARY COMPLIANCE'}
'''

train_df = pd.concat([train_df, pd.get_dummies(df.case_group),
                                pd.get_dummies(df.last_inspection_result)], axis=1)

train_df.columns = [ col.replace('/', '').replace('  ', '_').replace(' ', '_').lower() for col in new_cols ]

X = train_df[[u'citation', u'condo_conversion_and_presale', u'construction',
        u'housing_zoning', u'noisesound_levels', u'tenant_relocation_ordinance',
        u'unfit_building_or_premises', u'vacant_building_monitoring',
        u'building', u'building_and_premises', u'condocoop_conversion',
        u'construction_noise', u'electrical', u'housing',
        u'just_cause_eviction', u'mechanical', u'nonconstruction_noise',
        u'other_construction', u'premises', u'presale', u'signs', u'site',
        u'tenant_relocation_assist_ord', u'trao_avoidance', u'vacant_building',
        u'weeds_and_vegetation', u'zoning', u'failed', u'hold', u'passed']]

train_df['numeric_status'] = df.status.map(lambda x: status.index(x))
y = train_df['numeric_status']


X_train, X_test, y_train, y_test = train_test_split(X, y)

rf_model = RandomForestClassifier(n_estimators=1000)
rf_model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, rf_model.predict(X_test))
# score from train test split = 0.68390489755669992
