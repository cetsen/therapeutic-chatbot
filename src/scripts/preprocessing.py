import pandas as pd
import csv


if __name__ == "__main__":
    col_names = ["conversation id","subreddit","post title","author","dialog turn","text","compound","sentiment","emotion prediction"]
    df = pd.read_csv("data/RED/anxietyhelp_dyadic_convs_clean_emotion.csv", names=col_names, engine='python', dtype={'conversation id': 'unicode'})
    df = df.rename(columns={'conversation id': 'conversation_id', 'post title': 'post_title', 'dialog turn': 'dialog_turn', 'emotion prediction': 'emotion_prediction'})
    print("Number of conversations in raw dataset: ", df['conversation_id'].nunique())
    print("Number of rows in raw dataset: ", len(df))

    # Drop rows which have same text and conversation_id
    df = df.drop_duplicates(subset=['text','conversation_id'],keep="first")
    print("Number of rows after duplicates are dropped: ", len(df))

    # Group data by conversation id and calculate count of each conversation id
    df_conv_len = df.groupby("conversation_id").size().reset_index(name='num_dialog_turns')

    # Separate conversation id's with multiple occurrences as dialogues
    df_dia = df_conv_len[df_conv_len["num_dialog_turns"] > 2]
    print("Number of conversations longer than 2 turns: ", len(df_dia))
    df_dia = df_dia.reset_index()
    df_dia = df_dia.drop(columns=['num_dialog_turns'])

    # Only conversations with multiple turns remain in the dataset
    df = df.join(df_dia.set_index('conversation_id'), on='conversation_id', how="right") 

    # Separate conversations that have exactly 2 authors
    df_conv_authors = df.groupby("conversation_id")["author"].unique().reset_index()
    df_conv_authors["author"] = df_conv_authors["author"].apply(lambda x: x.size)
    df_conv_authors = df_conv_authors[df_conv_authors["author"] == 2]
    df_conv_authors = df_conv_authors.drop(columns=['author'])

    # Only conversations that have 2 authors remain in the dataset
    df = df.join(df_conv_authors.set_index('conversation_id'), on='conversation_id', how="right") 
    print("Number of conversations longer than 2 turns with 2 authors: ", len(df_conv_authors))
    df.reset_index(drop=True, inplace=True)

    # Number of turns of each author per conversation id
    df_num_author_turns = df.groupby(['conversation_id','author']).size().reset_index(name="author_num_turns")

    # Speakers of each conversation with their number of speaker turns
    df_speakers = df.groupby('conversation_id').first()['author'].reset_index(name='author')
    df_num_speaker_turns = pd.merge(df_speakers, df_num_author_turns)
    df_num_speaker_turns = df_num_speaker_turns[df_num_speaker_turns['author_num_turns'] > 1]
    df_num_speaker_turns = df_num_speaker_turns['conversation_id']
    df_num_speaker_turns = df_num_speaker_turns.drop(columns=['author','author_num_turns'])

    df = pd.merge(df, df_num_speaker_turns, on="conversation_id")
    print("Number of conversations longer than 2 turns with 2 authors and with multiple speaker turns: ", df['conversation_id'].nunique())

    df = df.dropna(subset=['compound','sentiment','emotion_prediction'])
    print("Number of conversations in cleaned dataset: ", df['conversation_id'].nunique())
    print("Number of rows in cleaned df: ", len(df))

    df.drop(columns="index", inplace=True)
    df.to_csv("data/RED/clean/anxietyhelp_clean.csv", index=False)
    print("Preprocessed dataset succesfully written into file")