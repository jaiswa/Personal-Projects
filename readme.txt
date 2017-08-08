All codes have been written using ipython notebook .Problem no. 1 and 2 are solved.

For problem number 1 :

Introduction and procedure:
Positive and negative reviews need to be detected based on the training data set. The training data provided has been split into 1900  training and 100 test set . Data set and training set are converted to vectors using CountVectorizer and TfidfTransformer . Multinomial naivebayes has been used for training.Results have been saved in CSV file named prob1_results

Requirements:
Path1 and path2 need to be set for the folder containing positive reviews and negative reviews respectively.
Pandas dataframe is used.

For Problem 2 :

Introduction and procedure:
Book Description and titles needed to be matched so the Tf-Idf of a title’s all words in all the descriptions and in the title itself were found out and then for each of  the title the cos similarity of its words’ Tf-Idf is found with all the descriptions Tf-Idf and the the description with which its cos similarity is maximum has that title.

Requirements :
Input need to be given in a file named ‘prob1_input.txt’   in the format given below and path need to be set accordingly also I have attached one input file:

An Integer N on the first line. This is followed by 2N+1 lines.
Text fragments (numbered 1 to N) from Set A, each on a new line (so a total of N lines).
A separator with five asterisk marks "*" which indicates the end of Set A and the start of Set B.
Text fragments (numbered 1 to N) from Set B, each on a new line (so a total of N lines). 


5
How to Be a Domestic Goddess : Baking and the Art of Comfort Cooking (Paperback)
Embedded / Real-Time Systems 1st Edition (Paperback)
The Merchant of Venice (Paperback)
Lose a Kilo a Week (Paperback)
Few Things Left Unsaid (Paperback)
*****
Today the embedded systems are ubiquitous in occurrence, most significant in function and project an absolutely promising picture of developments in the near future.
The Merchant Of Venice is one of Shakespeare's best known plays.
How To Be A Domestic Goddess: Baking and the Art of Comfort Cooking is a bestselling cookbook by the famous chef Nigella Lawson who aims to introduce the art of baking through text with an emphasis.
Lose A Kilo A Week is a detailed diet and weight loss plan, and also shows how to maintain the ideal weight after reaching it.
Few Things Left Unsaid is a story of love, romance, and heartbreak.
