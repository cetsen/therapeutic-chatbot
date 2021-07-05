# What Makes a Conversation Satisfying and Engaging? – An Analysis on Reddit Distress Dialogues

## Abstract
Recently, AI-driven chatbots have gained interest to help people deal with emotional distress and help them regulate emotion. However, since conversational data between patients who are in emotional distress and therapists who are actively offering emotional support is hardly available publicly due to privacy and ethical reasons, the most feasible option is to train chatbots on data from online forums such as Reddit. One challenge is ensuring that the data collected from these platforms contain responses that lead to high engagement and satisfaction and avoid those that lead to dissatisfaction and disengagement.  

We have developed a novel scoring function that can measure the level of satisfaction and engagement in distress oriented conversations. Using this scoring function, we classified dialogues in the Reddit Emotional Distress (RED) dataset as highly satisfying, less satisfying, highly engaging, and less engaging. By analysing these separate dialogues, we finally came up with a set of guidelines that describes which conversational strategies lead to highly satisfying and highly engaging conversations and which conversational strategies lead to less satisfying and less engaging conversations. Our guidelines can serve as a set of rules when developing therapeutic chatbots from online mental health community data so that inappropriate responses could be avoided and speaker satisfaction and engagement with these chatbots could be increased. 

## Dataset
The dataset can be found in the following folders in [Google Drive](https://drive.google.com/drive/folders/1Fg5RvwlGQ5s1k3YHmzkkg9-f3d77hMD0?usp=sharing):
* *RED/100_annotated*: 100 dialogues picked randomly from each subreddit of the RED dataset, annotated with their ground truth labels of engagement and satisfaction
* *RED/clean/unlabeled*: Clean dyadic conversations from the RED dataset with their sentiment and emotion predictions
* *RED/clean/labeled*: Clean dyadic conversations from the RED dataset with their sentiment, emotion, engagement, and satisfaction predictions
* *sarcasm*: News headlines dataset for sarcasm detection
