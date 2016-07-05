# Data Science Fellow

I chose to tackle the modeling portion of the problem.

### Approach

This was a very short amount of time to make any good insights.  My approach was to make dummies for all of the relevant data and try to predict on the status column of the dataset.  

Everything was completed in Python 2.7 using SKLearn.

### Conclusion
This was a classification problem and my final accuracy score was .6839 out of 1.  The most important features in my model from most import to least important were:

```python
[(u'passed', 0.21535078755951043),
 (u'failed', 0.1629174783752311),
 (u'citation', 0.1268775031352638),
 (u'weeds_and_vegetation', 0.12039954980826809),
 (u'housing_zoning', 0.0852814145759413),
 (u'housing', 0.0746205671836776),
 (u'zoning', 0.03655966258114758),
 (u'vacant_building', 0.02550633085387531),
 (u'tenant_relocation_ordinance', 0.024131977120125575),
 (u'construction', 0.01865865278989484),
 (u'construction_noise', 0.016737362555876233),
 (u'just_cause_eviction', 0.016655817388347798),
 (u'vacant_building_monitoring', 0.01550996876056416),
 (u'unfit_building_or_premises', 0.010736633865577586),
 (u'tenant_relocation_assist_ord', 0.009348263978115756),
 (u'building_and_premises', 0.007973864651484044),
 (u'building', 0.007586319188234977),
 (u'condocoop_conversion', 0.004169477372713437),
 (u'condo_conversion_and_presale', 0.004110769823529012),
 (u'site', 0.003888369049423176),
 (u'nonconstruction_noise', 0.0034448549270350856),
 (u'signs', 0.002901241040090177),
 (u'hold', 0.0019710791183182165),
 (u'electrical', 0.0008954317711771536),
 (u'trao_avoidance', 0.0008717702064093769),
 (u'other_construction', 0.0008143098945721706),
 (u'mechanical', 0.0008058749070737938),
 (u'noisesound_levels', 0.0007581859593465406),
 (u'premises', 0.0003417678956625009),
 (u'presale', 0.00017471366351279915)]
```

In the end my model did learn something from the data I fed into it.  The most important features were where they passed or failed.  I would have liked more time to do some EDA on the data to see how each feature correlated with the other features.  I feel like I could have done a better job of feature engineering with more time as well.  Additionally running a Grid Search on my parameters and Cross Validating the data would have been nice.
