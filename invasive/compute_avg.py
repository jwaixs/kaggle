import sys

import pandas as pd
import numpy as np

submission_file = sys.argv[1]
submission_file2 = sys.argv[2]
output_file = sys.argv[3]

submission_files = sys.argv[1:-1]
output_file = sys.argv[-1]

print submission_files, output_file

submissions = [pd.read_csv(csv_file).fillna(1.0) for csv_file in submission_files]

prediction = pd.DataFrame()
prediction['name'] = submissions[0]['name']
prediction['invasive'] = sum([sub['invasive'] for sub in submissions]) / len(submissions)
prediction.to_csv(output_file, index = False)

#f0 = pd.read_csv('./{}_fold_0.csv'.format(submission_file))
#f1 = pd.read_csv('./{}_fold_1.csv'.format(submission_file))
#f2 = pd.read_csv('./{}_fold_2.csv'.format(submission_file))
#f3 = pd.read_csv('./{}_fold_3.csv'.format(submission_file))
#f4 = pd.read_csv('./{}_fold_0.csv'.format(submission_file2))
#f5 = pd.read_csv('./{}_fold_1.csv'.format(submission_file2))
#f6 = pd.read_csv('./{}_fold_2.csv'.format(submission_file2))
#f7 = pd.read_csv('./{}_fold_3.csv'.format(submission_file2))
#
##median = list()
##for i in range(f0.shape[0]):
##    e0 = f0.ix[i].invasive
##    e1 = f1.ix[i].invasive
##    e2 = f2.ix[i].invasive
##    e3 = f3.ix[i].invasive
##    median.append(np.median([e0, e1, e2, e3]))
#
#prediction = pd.DataFrame()
#prediction['name'] = f0['name']
#prediction['invasive'] = (f0['invasive'] + f1['invasive'] + f2['invasive'] + f3['invasive'] + f4['invasive'] + f5['invasive'] + f6['invasive'] + f7['invasive']) / 8
#prediction.to_csv(output_file, index = False)
#
##f0['invasive'] = (f0['invasive'] + f1['invasive'] + f2['invasive'] + f3['invasive']) / 4
##f0.to_csv('./20170514_submission_test_densenet_avg.csv', index = False)
