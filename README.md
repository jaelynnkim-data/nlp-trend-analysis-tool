# [NLP-Based Industry Trend Analysis Tool Using Earnings Call Transcripts](https://github.com/jaelynnkim-data/nlp-trend-analysis-tool/blob/main/NLP_Trend_Analysis_Code.ipynb)

### Preface
Tilia Holdings, a private equity firm specializing in investments in food manufacturing
and distribution, was relying heavily on market research to guide investment decisions and enhance the
value propositions of their portfolio companies. Despite the critical nature of this research, Tilia
did not have an established approach to systematically gather and analyze research data. This
often resulted in significant time spent in manually reading earnings call transcripts and investing in consulting with industry
professionals. Manually eviewing eight years' worth of transcripts from just ten companies took a single employee at Tilia 
approximately 26 working days dedicated solely to reading. Even after
this extensive effort, the memory retention of the information and accuracy of their summary was not guaranteed.
To address these challenges, this project leveraged advanced natural language processing
(NLP) techniques to extract valuable insights from the earnings call transcripts of major public
companies in the food industry. This NLP system streamlined the analysis of these
transcripts, allowing Tilia to summarize strategic trends and identify key
market conditions relevant to senior leaders in the food space. The tool not only provided
Tilia with a deeper understanding of growth initiatives but also offered valuable context for its
existing portfolio companies, thereby enhancing the firm’s ability to make informed investment
decisions.


[UI Image 1](https://github.com/jaelynnkim-data/nlp-trend-analysis-tool/blob/main/UI%20Image%201.png)
[UI Image 2](https://github.com/jaelynnkim-data/nlp-trend-analysis-tool/blob/main/UI%20Image%202.png)


### Analysis Goals

The primary goal of this project was to equip Tilia Holdings with a powerful tool that
leverages Natural Language Processing (NLP) and machine learning techniques to analyze
earnings call transcripts from major food industry companies. This tool aimed to enhance Tilia’s
ability to make data-driven investment decisions, ultimately leading to improved investment
performance and market positioning. Specifically, the project focused on the following key
objectives:
1. Strategic Focus Extraction: Automatically identify and extract key strategic themes
from the text within public earnings call transcripts.
2. Sentiment Analysis: Analyze the tone and direction of company communications
associated with each strategic focus areas.
3. Trend Analysis: Track and visualize trends in both strategic focus and sentiment over
time.

The insights generated from this analysis were then integrated into an interactive, user-friendly tool that
allowed Tilia to efficiently explore and segment the data. The tool includes the following features:
1. Filtering capabilities that allow for filtering on transcript release dates, strategic focus
categories, and sentiment types.
2. A list of key topics and trends associated with the filtered subset of data
3. A term search bar
4. A visual depiction of patterns and movements of topics and sentiment


### Data Source
The input dataset comprises of earnings call transcripts from Capital IQ, a widely trusted platform
in the investment community for financial data and analysis. These transcripts cover major
players in the food industry over multiple years and financial quarters, providing detailed
insights into financial performance, strategic initiatives, and management’s views on market
conditions. The full dataset consisted of 300 PDF transcripts, each linked to a specific company’s earnings
call. Key metadata such as company name, year, and quarter were carefully extracted from the
filenames, ensuring a comprehensive view of the sector. An example transcript is provided in
Appendix A.
Capital IQ was chosen for its reliability and the depth of its data, which our client already
uses for accessing earnings call transcripts. The platform’s consistent formatting and high-quality
information make it ideal for this analysis, ensuring the accuracy and relevance of financial
discussions. By using Capital IQ, we can be confident that the data reflects the most pertinent
financial conversations, providing a solid foundation for our analysis. This choice supports the
overall integrity and reliability of the insights generated from the dataset.

### Feature Engineering
The feature engineering process focused on extracting meaningful information from
earnings call transcripts of major food companies. With each transcript from each company
containing different copyright notices and varying formats of PDF remnants, it was a priority to
develop a code that would remove any irrelevant text, such as copyright notices and disclaimers,
no matter the name of the company or the unexpected variants of watermarks. After the text
cleaning process, a data frame containing the company name, transcript release year and quarter as well as groups of 20 sentences from each transcript in each row was developed. The sentence
groups were then given as input to a Transformer-based summarization model to be treated as
paragraphs, where they were then transformed into three to eight sentences. The summarized
sentences were separated into individual sentences for each row and used as input for the rulebased topic detection model and the sentiment analysis model. These individual sentences were
then further processed into n-grams ranging from 1 to 5 words. These n-grams were filtered
based on their part-of-speech (POS) tags to retain only the most informative grammatic
sequences, which are more likely to carry strategic insights, resulting in a collection of words
and terms that could potentially be the trend keywords of the text. Additionally, custom filtering
rules were applied to remove company-specific words, abstract nouns, non-informative terms,
and overly generic phrases. The n-grams were also checked for redundancy using a Word2Vec
model to ensure that only unique and meaningful n-grams were retained. This resulted in a data
frame containing the company name, transcript year, transcript quarter, sentence, main topic,
sub-topic, sentiment label, sentiment score and keywords from each transcript.


### Modeling Framework

The modeling framework employed a robust combination of Natural Language
Processing (NLP) techniques and machine learning models to achieve three primary objectives:
Keyword Extraction, Topic Detection, and Sentiment Analysis. Each model was designed with
specific goals in mind and required a tailored approach to address the challenges presented by
the unstructured nature of the data


#### Keyword Extraction
For Keyword Extraction, the goal was to identify emerging buzzwords and technology
terms that users might not have previously encountered. The unpredictability of unstructured text
data required the model to handle noise effectively while accurately identifying contextually significant keywords, which often varied in length and consisted of word combinations forming
unique terms. 


To tackle this challenge, the system used a T5-base summarization model. This model
was selected after extensive testing of alternatives, including PEGASUS, BART, and
DistilBART. These tools are all part of a category known as transformers, which are cutting-edge
natural language processing (NLP) technologies designed to understand and generate human
language. Transformers work by processing text in chunks, paying attention to the context of
each word relative to the others in the text. This is done using a “self-attention” mechanism,
which allows the model to weigh the importance of each word depending on the context in which
it appears. 


T5-base was chosen for its ability to balance computational efficiency with the precision
of summarization, capturing both the main message and supporting anecdotes from the text.
After summarization, the sentences were transformed into n-grams ranging from one to five
words, each tagged using Part-of-Speech (PoS) tagging. This process enabled the system to filter
out irrelevant terms and focus on retaining potentially trending keywords.


One of the key assumptions in the keyword extraction process was that meaningful
keywords and phrases could be effectively identified through n-gram analysis and PoS tagging.
The model assumed that the context and significance of these terms could be preserved in the
summarized text and that the inherent noise in unstructured text could be sufficiently filtered out.
To achieve optimal performance, the summarization model was fine-tuned using a corpus of
earnings call transcripts, with hyperparameters adjusted to balance processing time and accuracy.
The n-gram model was parameterized to focus on sequences of 1 to 5 words, with filtering rules
iteratively refined to remove irrelevant and redundant terms. The performance of the model was validated by comparing the extracted keywords against known industry terms and emerging
trends, ensuring that the output was both relevant and actionable.


#### Topic Detection
In the case of Topic Detection, the objective was to categorize text according to specific
investment topics of interest to Tilia. This component was crucial for providing a deeper
contextual understanding of the data by mapping sentences to predefined topics that reflect
business value and technological advancements. The approach involved creating a handcrafted
dictionary comprising approximately 400 topic assignments, manually annotated to align with
Tilia's investment interests. This dictionary served as the foundation for training a topic labeling
model, which assigned one of the 18 predefined topics to each summarized sentence. The model
relied on traditional text classification methods, comparing each sentence against the annotated
dictionary to determine the best topic match.


The primary assumption underlying this approach was that the handcrafted dictionary
was comprehensive enough to cover the range of topics relevant to Tilia's investment strategy. It
also assumed that the manual annotations were accurate and representative of the broader
dataset, and that the model could generalize effectively from the training data to new, unseen
sentences in the earnings call transcripts. The model was trained on the annotated dictionary
using standard text classification techniques, and its performance was validated through a
confusion matrix. This matrix compared the model’s predictions against a test set of sentences
manually labeled by domain experts. The evaluation highlighted the model's ability to
distinguish between closely related topics and informed further refinements to improve accuracy.


#### Sentiment Analysis
Sentiment Analysis played a crucial role in evaluating the sentiment behind detected
topics within the context of the food industry's earnings call transcripts. This component was
essential for understanding how market conditions, company strategies, and other factors were perceived by industry players. Initially, the sentiment analysis model was based on the
ProsusAI/finbert model, fine-tuned using the Takala/financial_phrasebank dataset. However, the
model faced challenges in distinguishing between positive and neutral sentiments, particularly in
sentences containing both strongly positive and strongly negative phrases. To address this, a
sentence compression model was integrated into the process. This model, built on a T5-small
architecture, distilled sentences to their core meanings, removing extraneous language that could
skew sentiment analysis. The final sentiment analysis was conducted using a RoBERTa-based
model, which proved more effective after incorporating the sentence compression step.


The sentiment analysis process was based on the assumption that reducing sentences to
their essential meanings through compression would improve sentiment classification accuracy.
The model also assumed that the fine-tuning dataset (Takala/financial_phrasebank) was
representative of the language and sentiment typically found in earnings call transcripts. The
initial sentiment model was trained and fine-tuned using 4,000 rows of financial expert
annotations. However, after identifying its limitations, the model was retrained with the sentence
compression step integrated into the process. The RoBERTa model was further fine-tuned using
the same dataset, with parameters adjusted to optimize sentiment classification accuracy.
Validation involved testing the model on a hand-labeled dataset of earnings call sentences,
confirming that the sentence compression improved the model’s performance, particularly in
distinguishing between positive and neutral sentiments.









