= Toxic Comment Classification Challenge
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

== Description
Discussing things you care about can be difficult. The threat of abuse and
harassment online means that many people stop expressing themselves and give up
on seeking different opinions. Platforms struggle to effectively facilitate
conversations, leading many communities to limit or completely shut down user
comments.

The Conversation AI team, a research initiative founded by Jigsaw and Google
(both a part of Alphabet) are working on tools to help improve online
conversation. One area of focus is the study of negative online behaviors, like
toxic comments (i.e. comments that are rude, disrespectful or otherwise likely
to make someone leave a discussion). So far they’ve built a range of publicly
available models served through the Perspective API, including toxicity. But
the current models still make errors, and they don’t allow users to select
which types of toxicity they’re interested in finding (e.g. some platforms may
be fine with profanity, but not with other types of toxic content).

In this competition, you’re challenged to build a multi-headed model that’s
capable of detecting different types of of toxicity like threats, obscenity,
insults, and identity-based hate better than Perspective’s current models.
You’ll be using a dataset of comments from Wikipedia’s talk page edits.
Improvements to the current model will hopefully help online discussion become
more productive and respectful.

Disclaimer: the dataset for this competition contains text that may be
considered profane, vulgar, or offensive.

== Evaluation
Submissions are evaluated on the mean column-wise log loss. In other words, the
score is the average of the log loss of each predicted column.  Submission File

For each id in the test set, you must predict a probability for each of the six
possible types of comment toxicity (toxic, severe_toxic, obscene, threat,
insult, identity_hate). The columns must be in the same order as shown below.
The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
6044863,0.5,0.5,0.5,0.5,0.5,0.5
6102620,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

== Prizes
* 1st Place - $18,000
* 2nd Place - $12,000
* 3rd Place - $5,000

== Timeline
* February 13, 2018 - Entry deadline. You must accept the competition rules
  before this date in order to compete.
* February 13, 2018 - Team Merger deadline. This is the last day participants
  may join or merge teams.
* February 20, 2018 - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise
noted. The competition organizers reserve the right to update the contest
timeline if they deem it necessary.

== Kernels
* NB-SVM strong linear baseline - https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline-eda-0-052-lb

== Literature
* https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf
* https://en.wikipedia.org/wiki/Tf%E2%80%93idf
