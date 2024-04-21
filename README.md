The research aims to
 identify the most effective technique for interpreting customer feedback across various diverse digital communication channels and provide to customers a useful tool through which they can extract important information for products or interest.

The main code is in handle_booking_reviews_up_vader_csv_git.py file and can be run via a relevant python tool like idle python. 
The data set used is booking.com reviews (Hotel_Reviews.csv) which should be localy stored and can be downloaded from internet , for example from https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe

There is also another file (pexels-pixabay-87651.jpg) not important for the final results but is only for the presentation of UI , but it should also be locally downloaded. 

So there are two points in the code that should be modified in order to be prorerly run. 

When UI opens there is a free test bar , where you can write parts of the hotel name for which you need to get ifno based on sentiment analyses of reviews included in Hotel_Reviews.csv

After selecting the hotel and from the drop down menu ,you can select the info needed. 
 - View graphs
The user interface gives to the final user to view several graphs with useful visualized information about the hotel that was chosen (frequency of words of interest and categories, total number of apprearence of these words in negative, positive of neutra reviews, would cloud)
-Check reviews and tags
When user selects this option a new graphs appears that show the relationship between hotel, reviews and tags. For that a neo4j graph database used
- View all reviews
show all the reviews of the specific hotel with no further analyses or cleanness. There are the raw data of the available already posted in booking.com reviews.
- View positive reviews
This selection allows users to take a lot only in positive reviews tagged by VADER method. User can check whether the reviews are similar and the most positive benefits of the hotel.
- View negative reviews
Similarly to previous option with ‘view negative reviews’ user is able to check only the available negative reviews of the selected hotel. Obviously he can check the disadvantages of the hotel and decide how import these are for him.
- View sum up of reviews
The last option that is available to users is ‘view sum up of the reviews’ which returns a pop up windows with the summary of all the reviews of the selected by the user hotel.
