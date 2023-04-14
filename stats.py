import pandas as pd

icd10_to_10am_df = pd.read_csv('results.csv', header=None)
icd10am_to_10_df = pd.read_csv('results_rev.csv', header=None)


## icd 10 to 10am
## code to code
correct_10_to_10am = 0

## df format [ [input] [gt] [predicted]]
for i in range(len(icd10_to_10am_df)):
        if icd10_to_10am_df.iloc[i, 1] == icd10_to_10am_df.iloc[i, 2]:
            correct_10_to_10am = correct_10_to_10am + 1

print(f'Accuracy (ICD10 to ICD10AM): {correct_10_to_10am/len(icd10_to_10am_df)}')

### check three digits 
correct_10am_3 = 0
for i in range(len(icd10_to_10am_df)):
        if icd10_to_10am_df.iloc[i, 1] == icd10_to_10am_df.iloc[i, 2] or icd10_to_10am_df.iloc[i, 1][0:3] == icd10_to_10am_df.iloc[i, 2][0:3]:
            correct_10am_3 = correct_10am_3 + 1
print(f'Accuracy (ICD10 to ICD10AM) upto three digits: {correct_10am_3/len(icd10_to_10am_df)}\n')


## icd 10am to 10
## code to code
correct_10am_to_10 = 0

## df format [ [input] [gt] [predicted]]
for i in range(len(icd10am_to_10_df)):
        if icd10am_to_10_df.iloc[i, 1] == icd10am_to_10_df.iloc[i, 2]:
            correct_10am_to_10 = correct_10am_to_10 + 1

print(f'Accuracy (ICD10AM to ICD10): {correct_10am_to_10/len(icd10am_to_10_df)}')

### check three digits 
correct_10_3 = 0
for i in range(len(icd10am_to_10_df)):
        if icd10am_to_10_df.iloc[i, 1] == icd10am_to_10_df.iloc[i, 2] or icd10am_to_10_df.iloc[i, 1][0:3] == icd10am_to_10_df.iloc[i, 2][0:3]:
            correct_10_3 = correct_10_3 + 1
print(f'Accuracy (ICD10AM to ICD10) upto three digits: {correct_10_3/len(icd10am_to_10_df)}')
