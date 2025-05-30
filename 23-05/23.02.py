#Loan Approval
import pandas as pd

df = pd.read_csv("23-05/loan_approval_dataset.csv")

#Total number of records
N = len(df)

# A = Approved
A = len(df[df['loan_status'] == 'Approved'])

#B = Good Credit (CIBIL >= 700)
B = len(df[df['cibil_score'] >= 700])

#A and B = Approved and Good Credit
A_and_B = len(df[(df['loan_status'] == 'Approved') & (df['cibil_score'] >= 700)])
P_A = A / N    #P(Loan Approved)
P_B = B / N    #P(Good Credit)
P_B_given_A = A_and_B / A   #P(Good Credit | Loan Approved)

#Bayes' Theorem
P_A_given_B = (P_B_given_A * P_A) / P_B

print("Total Records: ", N)
print("Approved Count: ", A)
print("Good Credit Count: ", B)
print("Approved and Good Credit Count: ", A_and_B)
print("P(Loan Approved): ", P_A)
print("P(Good CIBIL Score): ", P_B)
print("P(Good CIBIL Score | Loan Approved): ", P_B_given_A)
print("P(Loan Approved | Good Credit): ", P_A_given_B)

#Two new customers walk in, one with a CIBIL score of 800 and another with a CIBIL score of 500.
#Bad Credit count
bad_credit = N - B  #People with CIBIL score less than 700

#Appoved and Bad Credit
A_and_notB = A - A_and_B  #People who got approved but don't have good credit

# P(Loan Approved | Bad Credit)
P_A_given_notB = (A_and_notB / A) * (P_A / (bad_credit / N) )

print("P(Loan Approved | CIBIL Score = 800): ", P_A_given_B)  #Good Credit
print("P(Loan Approved | CIBIL Score = 500): ", P_A_given_notB)  #Bad Credit

#Now suppose out of 300 new custmers with bad credit rating, 285 were approved... This was because of some policy change

new_bad_credit = 300
new_approved_bad_credit = 285

#Update counts
bad_credit += new_bad_credit
A += new_approved_bad_credit
N += new_bad_credit  #Only new were bad credit, so total increases

A_and_notB += new_approved_bad_credit

#Recalculate probabilities
P_A = A / N   
P_notB = bad_credit / N
P_A_and_notB = A_and_notB / N
P_A_given_notB = P_A_and_notB / P_notB

print("After updating with 300 new bad credit customers: ")
print("P(Loan Approved | CIBIL Score = bad): {P_A_given_notB:.3f}")
