#Confusion matrix - Exercise

'''
Out of 1000 emails, 800 non-spams were classified correctly, 20 were incorrectly classified as spam, 
and 40 were incorrectly classified as non-spam, and remaining spams were classified correctly.

Write null hypotesis. and create a table of the confusion matrix.
'''
# Null hypothesis
null_hypothesis = "The classifier is not effective in distinguishing between spam and non-spam emails."
print(null_hypothesis)
# Confusion matrix table
print("Confusion Matrix:")
print("  | Predicted Spam | Predicted Non-Spam |")
print("-------------------------------------------------")
print("Actual Spam |", 180, "|", 20, "|")
print("Actual Non-Spam |", 40, "|", 760, "|")
# Conclusion
print("Conclusion:")
print("The classifier is not effective in distinguishing between spam and non-spam emails.")