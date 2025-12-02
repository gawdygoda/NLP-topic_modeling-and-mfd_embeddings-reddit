import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "../data"
DATA_FILE = f"{DATA_DIR}/aita_filtered.pkl"
RESULTS_DIR = "../results"


# ------------------------------
# Start Time
# ------------------------------
start_time = time.time()

# ------------------------------
# Read in dataframe
# ------------------------------
raw_data = pd.read_pickle(DATA_FILE)

# ------------------------------
# Perform EDA
# ------------------------------
print("The Dataset looks like:")
print(raw_data.head())
print(raw_data.shape)
print("====================================")

# Set options to show all columns of the dataset
pd.set_option("display.max_columns", None)
# Display all the columns together in the console
print("Display first 5 rows")
print(raw_data.head().to_string())
print("====================================")
print("Basic Dataframe info")
print(raw_data.info())
print("====================================")
print("More detailed Dataframe info")
print(raw_data.describe(include="all").to_string())
print("====================================")
print("Number of Empty values in each column:")
print(raw_data.isnull().sum().sort_values(ascending = False))
print("====================================")
print("Number of Unique values in each column:")
print(raw_data.apply(pd.Series.nunique))
print("====================================")
print("Are there duplicate rows?")
print(raw_data.duplicated().sum())
dup_all = raw_data[raw_data.duplicated(keep=False)]
print(dup_all)
print("====================================")
print("Unique values in 'resolved_verdicts':")
rv_counts = raw_data["resolved_verdict"].value_counts()
print(rv_counts)
print("====================================")
num_words = raw_data["num_words"]
max_words = raw_data["num_words"].max()
over_512 = (raw_data["num_words"] > 512).sum()
print("Max num_words:", max_words)
print("Count of rows where num_words > 512:", over_512)
print("====================================")

# ------------------------------
# Plotting
# ------------------------------

#Plot Resolved Verdict Pie Graph
plt.figure(figsize=(6,6))
plt.pie(rv_counts.values, labels=rv_counts.index, autopct="%1.1f%%", startangle=90)
plt.title("Distribution of Resolved Verdicts")
filename = f'{RESULTS_DIR}/ResolvedVerdictsDistribution.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {filename}")


# Plot Num Words Histogram
plt.figure(figsize=(10, 6))
sns.histplot(num_words, bins=60, kde=False)
plt.axvline(512, color='red', linestyle='--', linewidth=2, label="~512-token limit")
percentage = over_512/len(raw_data)*100
plt.text(
    512 + 20,                     # Slightly to the right of the line
    plt.ylim()[1] * 0.9,          # 90% of the plot height
    f"{over_512:,}({percentage:.0f}%) posts > 512 words",
    color='red',
    fontsize=12
)
plt.title(f"Histogram of Post Word Counts\nPost with Max words = {max_words:,}\nNote: Word count is a rough proxy for BERT’s token limit")
plt.xscale("log")
plt.xlabel("Number of Words")
plt.ylabel("Number of Posts")
plt.legend()
plt.tight_layout()
filename = f'{RESULTS_DIR}/NumWordsHistogram.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {filename}")

# ------------------------------
# End Time
# ------------------------------
end_time = time.time()
elapsed = end_time - start_time
mins, secs = divmod(elapsed, 60)

print(f"Total time: {mins:.0f} min {secs:.2f} sec")




# The Dataset looks like:
#             author                                              title  ... num_words  resolved_verdict
# 0  theworldbystorm  AITA Had a disagreement about Les Miserables w...  ...       227               YTA
# 1          pfoffie      AItA because I sent rude SMS to an ex-friend?  ...       331               YTA
# 2        [deleted]         AItA for getting yelling at my friends GF?  ...       518               NTA
# 3     notapantsday            AITA for selling an unreliable product?  ...       233               YTA
# 4        jugglesme  AITA for not being appreciative enough of the ...  ...       728               NTA

# [5 rows x 7 columns]
# (108634, 7)
# ====================================
# Display first 5 rows
#             author                                                            title                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           selftext  num_comments         created_utc  num_words resolved_verdict
# 0  theworldbystorm    AITA Had a disagreement about Les Miserables with a coworker.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            I love the musical *Les Miserables*. A coworker of mine said that they didn't like it. I asked why and they said, among other things, that "it's not a musical, it's an opera."\n\nNow, I happen to be an opera singer. I told them adamantly that it is NOT an opera. There are a number of things that make this true, but if you can take my word for it- and I hope you can- it's not a matter of opinion whether a mainstream musical is an opera or not. It is a musical, period.\n\nMy coworker thought otherwise. "My boyfriend writes musicals and he agrees with me" was the extent of her argument. Well, he may well write musicals but that doesn't mean he knows anything about opera. You might as well ask a jazz pianist to tell you about Beethoven- it's not the same area of expertise even if it's related. I told her "You're entitled to your opinion even if it's wrong" in a sort-of joking way. Maybe I was being dickish.\n\nI gotta know- am I the asshole?            15 2014-02-25 00:48:17        227              YTA
# 1          pfoffie                    AItA because I sent rude SMS to an ex-friend?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          There's a girl, once my best friend. She was always a bit clumsy, destroyed several things such as cups and drinking glasses. However I of course didn't mind because we were friends. I also didn't mind that several times I had an appointment with her and took a long trainride to get to her and she wasn't there. then she realized I had more than just one friend and became super jealous. She insulted my friends very badly, called me a bad friend when I went on holidays with someone else. And many other things. At some point our friendship died...\n\nSeveral months later, xmas 2012, we invited our families and needed 8 wine glasses. But we had just 7 .. because she had destroyed one. I was so angry because this made me remember her and what she did. So I sent her a very insulting SMS. Because I wanted her to feel bad on xmas too.\n\nAnother few months later, something else happened - I actually don't remember - that made me angry about her again. I sent another VERY insulting SMS.\n\nNow she sued me and won. I have to pay a more than 900$ penalty.\n\nShe's got borderline, which of course explains EVERYTHING she's done. I just wasn't capable of handling that anymore at some point.\n\nI feel like I had the "right" to insult her after all she's done and of course all my friends say I'm right. But I'm really worried that I am the actual asshole after all...\n\ntl;dr: Girl destroyed wine glass. Later we stopped being friends. Later I needed a wine glass and sent her an insulting SMS because it was missing. AItA?            12 2014-02-25 07:40:53        331              YTA
# 2        [deleted]                       AItA for getting yelling at my friends GF?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             Tonight I went to my friends house and we there was a bonfire and we were all having a good time. Then I made a comment about how me and my friend, E, have been best friends since 4th grade. She calls me out, and starts asking questions like "what is hid favorite ___" and it actually made me really uncomfortable that I didn't know most of this. I've been friends with him for years but we would always talk about other subjects and never got to know the favorites of each other.\n\nBut I digress, let's move on to when it gets worse...\n\nHe comes out with fireworks that detonate when you throw them. My truck was parked right next to where we all were. Its been in perfect condition for the past 8 years, and she thought it would be funny to start throwing the fireworks at [my truck (costs 40k, 2005 GMC SLT)](http://i.imgur.com/vCcIWV5.jpg). I told her to stop, and she kept going. I told her that I like to take care of my shit and don't want someone to come along and fuck up what I worked on for 8 years. She laughs and throws more. Then she takes the wrapper and stuffs it in a hole in the trunk for towing straps.\n\nWe all agreed to go out to a place. Everyone piles in another guys car, but when my friends gets in my car his GF jumps in and almost scuffs up my leather seats. Then she proceeds to take the lighter and "playfully" hit me along the drive. This is when I yelled at her. "I've already fucking told you, Im trying to keep this vehicle in a good condition and everything you're doing is ruining 8 years of fucking work. Respect my shit or get the fuck out!" He (he's a joker) sarcastically said "hey man, apologize to her for that" and he and I laughed nervously, then she started hitting me and said, yeah, apologize!" Fortunately, we just pulled into the parking lot, and as soon as they got out my truck, I drove out of the parking lot, frustrated.\n\nI got home about an hour ago and still thinking about this. There's scuff marks all over the passenger side of my vehicle now, and I feel like a butthurt kid for getting so angry about it, but I try to take care of my shit. I'm not sure if my friend is seriously pissed at me for this.\n\n\nEdit: turns out he wasn't being sarcastic. He thinks I should seriously apologize for slapping his GF on the hand when she kept hitting me.            22 2014-03-09 06:30:43        518              NTA
# 3     notapantsday                          AITA for selling an unreliable product?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          Hi everyone,\n\na while ago I bought a water filter that is used for hiking or other outdoor activities. Its purpose is to make dirty or contaminated water drinkable. The filter I bought was from a very well respected manufacturer (Katadyn if anyone is interested) but this specific model had some major design flaws. I sent in my filter because it had failed after a short time and got a brand new unit in return. Because I didn't really trust this filter anymore, I got one from a different brand and sold the exchanged one on a small trekking forum. That was years ago and I kind of forgot all about it. Until I stumbled upon a travel report where someone has trouble with his water filter and eventually he and his friend become sick and have to cancel their trip. He wrote that he bought the filter on the forums, so I checked and it was in fact the filter that I had sold to him...\n\nSo now I feel like an asshole because I knew the filter was shit and I still sold it. On the other hand, it was a brand new unit from a respected manufacturer that is still sold at most commercial outdoor stores.\n\nAm I the asshole here?            14 2014-03-17 21:19:08        233              YTA
# 4        jugglesme  AITA for not being appreciative enough of the help I was given?  Recently I spent a good amount of time traveling around New Zealand. I had a tiny budget, so I was getting around by hitch hiking and sleeping in my tent most nights. One afternoon I was picked up by a friendly older Kiwi. It turned out his business is consulting for struggling hostels, so he very frequently meets travelers like me and picks up hitch hikers. He has traveled the country dozens of times. On the way to my destination he went out of his way to take me to some cool thermal pools and pretty sights. It was great, because I likely would have just passed straight by them if anybody else had picked me up. \n\nHe was going on a speaking tour over the next week. Shortly before arriving at my destination he said, "It's too bad your not going to this other way with me next instead. I could show you this spot and that. And I could get you free accomodation in these towns." Since he had been so friendly, and because I wanted to be open to fun new experiences and people, I decided spur of the moment to change my plan and keep traveling with him instead. \n\nI stay in his hotel room with him that night. He got us both dinner with the food budget he had from his employer. He then suggested we buy some alcohol, and asks if I don't mind paying since he got dinner. I had planned on giving up drinking at this point in my trip to save money, but I was happy to since he was being so nice and helpful to me. I bought a couple bottles of wine that got polished off pretty fast. And he ended up sharing a lot of whiskey that he had brought, so he was definitely giving back more than he was asking. We probably stayed up till 2 in the morning drinking, and had some great conversation.\n\nThe next day we move on to the next city, and basically repeat the night before. I buy a couple bottles of wine, and he very insistently makes me take a couple whiskey shots. At this point his personality is really starting to grate on me, but I figure it would be rude to leave early when he is expecting me to keep traveling with him for a couple more days.\n\nDay 3, while we're driving he straight up calls me out for not being appreciative enough of him. In particular, he said that I should be making more of an effort to buy food or drinks. He was mad at me that he had to ask me to buy the wine, and that I wasn't pushing to do so myself. His reasoning is that I should be spending whatever money I am saving by not staying in a hostel on things that we can enjoy. He frequently hosts couch surfers, and says that is standard practice when somebody provides you with a place to stay. But I actually spent more money traveling with him than I would have without. And it seemed unfair to expect we should each contribute equally, because I had almost no money and he had kept bragging about how much money he was making from each of his speaking engagements. This pretty much spelled the end of traveling together for me, and I hitched another ride while he was in one of his speaking engagements. He never responded to my message when I told him what I was doing. \n\nEver since I've been worrying about this general situation. I'm very stingy with what I allow myself to buy. So I'm also not usually buying stuff that I share with other people. But when other people offer to share with me, I will often take it. I think I'm kind of a free loader. Where's the line where being frugal for myself becomes obnoxious to those around me?            11 2014-03-19 06:25:46        728              NTA
# ====================================
# Basic Dataframe info
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 108634 entries, 0 to 108633
# Data columns (total 7 columns):
#  #   Column            Non-Null Count   Dtype         
# ---  ------            --------------   -----         
#  0   author            108634 non-null  object        
#  1   title             108634 non-null  object        
#  2   selftext          108634 non-null  object        
#  3   num_comments      108634 non-null  int64         
#  4   created_utc       108634 non-null  datetime64[ns]
#  5   num_words         108634 non-null  int64         
#  6   resolved_verdict  108634 non-null  object        
# dtypes: datetime64[ns](1), int64(2), object(4)
# memory usage: 5.8+ MB
# None
# ====================================
# More detailed Dataframe info
#            author   title                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            selftext   num_comments                    created_utc      num_words resolved_verdict
# count      108634  108634                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              108634  108634.000000                         108634  108634.000000           108634
# unique      94145  107614                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              108624            NaN                            NaN            NaN                4
# top     [deleted]    AITA  He used to shit talk me to this girl months ago. \n\nShe said: I dunno what to say. I’m sad you’re not in my life (my fault) and I’m still curious and sad as to why I’m not enough. I wish I could have answers but I know you don’t want to be pressed (you said you didn’t wanna talk). So I guess goodbye then? I do Not think you’ll ever want to talk to me.\n\nHe: I'm extremely hesitant to but I would like to speak one day with you. There's a couple things I want to talk about. But if it's going to be a shit show or rehashing things, there's no reason.\n\nShe: It would mainly just be a final convo and I’m ok with that. Just kinda sucks\n\nHe: Unless one of us dies, I highly doubt that it would be the last convo ever. Maybe some time next week or the week after.\n\n            NaN                            NaN            NaN              NTA
# freq         2871      47                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   3            NaN                            NaN            NaN            63879
# mean          NaN     NaN                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 NaN      82.830099  2019-06-15 05:07:45.154380544     388.837307              NaN
# min           NaN     NaN                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 NaN      10.000000            2014-02-25 00:48:17      50.000000              NaN
# 25%           NaN     NaN                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 NaN      15.000000            2019-03-31 06:14:20     232.000000              NaN
# 50%           NaN     NaN                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 NaN      23.000000     2019-07-11 12:24:45.500000     354.000000              NaN
# 75%           NaN     NaN                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 NaN      44.000000     2019-10-05 22:59:52.500000     521.000000              NaN
# max           NaN     NaN                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 NaN   12533.000000            2020-04-04 17:58:02    6729.000000              NaN
# std           NaN     NaN                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 NaN     313.536475                            NaN     215.731328              NaN
# ====================================
# Number of Empty values in each column:
# author              0
# title               0
# selftext            0
# num_comments        0
# created_utc         0
# num_words           0
# resolved_verdict    0
# dtype: int64
# ====================================
# Number of Unique values in each column:
# author               94145
# title               107614
# selftext            108624
# num_comments          2004
# created_utc         108447
# num_words             1563
# resolved_verdict         4
# dtype: int64
# ====================================
# Are there duplicate rows?
# 0
# Empty DataFrame
# Columns: [author, title, selftext, num_comments, created_utc, num_words, resolved_verdict]
# Index: []
# ====================================
# Unique values in 'resolved_verdicts':
# resolved_verdict
# NTA    63879
# YTA    24295
# NAH    13688
# ESH     6772
# Name: count, dtype: int64
# ====================================
# Max num_words: 6729
# Count of rows where num_words > 512: 28289
# ====================================