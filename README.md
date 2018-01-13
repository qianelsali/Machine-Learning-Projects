# Data-Capstone-Projects

## 1. Project 1: Promotions towards Game Users at the Initial Installation Stage

Description of the project:

The majority of mobile game users never generate any revenue (i.e., ‘convert’, in marketing parlance). We want to offer a sale to some of these ‘non-payers’, in hopes of generating incremental revenue. Unfortunately, it’s not easy to identify true non-payers because many payers don’t convert until several weeks after installing the game. If you simply set a cutoff date and offer sales to everyone who hasn’t yet converted, you might find that revenue actually falls. This can occur if large numbers of not-yet-converted future payers take advantage of the sale instead of paying full price. If you’re not lucky, you might fail to convert a sufficient number of true non-payers, while ‘cannibalizing’ future payers by providing more value than they actually require.

The goal of this project is to design a sale that maximizes incremental revenue.

The data includes a group of users who installed during the first quarter of 2016.

    ●	User data, including user ID and install date
    ●	Session history, including date and session number
    ●	Purchase history, including date and amount
    ●	Spending history, including date, currency, and amount

Objectives: 

    a) explore gamer user behavior
    b) build a classfier to predict who are going to be payers and non-payers at the initial installation stage
    c) design a promotion on predicted payers and non-payers
    d) design metrics to evalute the promotion effects                  
 
 The framework of the code:
  
     0) explore the game users's behaviors:
         Active days, Convert Gap days, Pay Frequency of payers, Paid values, Averaged Paid values
     1) decide the time window
     2) select the target group
     3) feature selection
     4) modeling: logistic regression- sklearn
     5) test the classifer at the fourth day of installation
     6) implement the promotion
     7) evaluate promotion effects:
         revenue, 
         convert rate
         convert days
         active days
    


## 2. Project 2: prediction of New York Metro Station Flows

 Objectives: Using Time Series to estimate New York Metro Station flows
