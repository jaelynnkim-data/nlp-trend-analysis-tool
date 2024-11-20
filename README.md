# [NLP-Based Industry Trend Analysis Tool Using Earnings Call Transcripts](https://github.com/jaelynnkim-data/nlp-trend-analysis-tool/blob/main/NLP_Trend_Analysis_Code.ipynb)

### Preface
Tilia Holdings, a private equity firm specializing in investments in food manufacturing
and distribution, was relying heavily on market research to guide investment decisions and enhance the
value propositions of their portfolio companies. Despite the critical nature of this research, Tilia
did not have an established approach to systematically gather and analyze research data. This
often resulted in significant time spent in manually reading earnings call transcripts and investing in consulting with industry
professionals. Manually reviewing eight years' worth of transcripts from just ten companies took a single employee at Tilia 
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

![Intro Image](https://github.com/jaelynnkim-data/nlp-trend-analysis-tool/blob/main/Images%20for%20README/Intro_Image.webp)

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
2. Clickable box buttons containing key topics and trends associated with the filtered subset of data
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


![Earnings Call Transcript Images](https://github.com/jaelynnkim-data/nlp-trend-analysis-tool/blob/main/Images%20for%20README/Earnings%20Call%20Transcript%20Images.png)


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


![Process Flow Images](https://github.com/jaelynnkim-data/nlp-trend-analysis-tool/blob/main/Images%20for%20README/Process%20Flow%20Image.png)


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


#### Model Evaluation and Validation
The implementation of the modeling framework led to several important insights, along
with a thorough evaluation of the model's performance. The initial setup involved summarizing
groups of sentences using the DistilBART model, followed by sentence compression and
keyword extraction through Part-of-Speech (PoS) tagging. However, this approach proved
inefficient, producing outputs filled with nonsensical words, missing key trend keywords, and resulting in excessive processing time. These issues necessitated a reevaluation of each step to
ensure precision and computational efficiency.
To refine the summarization process, an experiment was conducted with different
sentence groupings, starting with 10 sentences per group and testing up to 50 sentences. The
optimal grouping was found to be 20 sentences, balancing computational efficiency and the
precision of summary results. This setup was validated through a manual comparison of
paragraphs and their summaries.
Further experimentation was carried out with four different summarization models—
PEGASUS, BART, DistilBART, and T5—to determine which provided the best balance of
speed and contextual accuracy. Through a combination of BLEU and ROUGE tests, T5-base
emerged as the superior model, outperforming the others across key metrics. The BLEU scores
were as follows: PEGASUS (0.0288), DistilBART (0.0344), BART (0.0388), and T5 (0.0475).
Similarly, T5-base showed higher ROUGE scores, with a ROUGE-1 score of 0.3118, compared
to BART’s 0.2892, DistilBART’s 0.2872, and PEGASUS’s 0.2487. These results confirmed that
T5-base was not only faster but also more adept at capturing both the main message and any
supporting anecdotes, leading to its selection as the final summarization model.


![Comparison of Summarization Model Performance](https://github.com/jaelynnkim-data/nlp-trend-analysis-tool/blob/main/Images%20for%20README/Comparison%20of%20Summarization%20Model%20Performance.png)


With the integration of T5-base, the sentence compression step became unnecessary for
keyword extraction since the summaries already contained concise and potentially relevant keywords. This allowed a direct focus on keyword extraction, where n-grams were employed to
capture multi-word terms that could not be recognized when isolated. PoS tagging was used to
ensure the extraction process retained only meaningful grammatical sequences, enhancing the
model’s ability to discard irrelevant n-grams.


![N-gram Filtering and Extraction from Summarized Sentences](https://github.com/jaelynnkim-data/nlp-trend-analysis-tool/blob/main/Images%20for%20README/N-gram%20Filtering%20and%20Extraction%20from%20Summarized%20Sentences.png)


In the keyword extraction process, an experiment was conducted with n-grams of various
lengths, ranging from 1 to 7 words, to determine the optimal range for capturing meaningful
keywords and terms. It was observed that n-grams longer than 5 words often resulted in phrases
that were too lengthy and convoluted to convey concise and meaningful information. The smaller
n-grams, particularly those ranging from 1 to 5 words, were sufficient to capture most of the key
information within sentences. Consequently, the decision was made to limit the extraction to 1-
to 5-grams, ensuring that the keywords generated were both precise and contextually relevant.


The topic detection model was evaluated using a confusion matrix, comparing predicted
topic labels against a manually labeled test set. The results revealed a high level of accuracy
(0.9076), with strong performance in identifying topics such as "Food Safety" and "Divestiture." However, the model did show some misclassifications, such as confusing "Sustainability" with
"Investment" and "Packaging." These errors highlighted areas for potential improvement,
particularly in distinguishing between closely related topics. The confusion matrix provided
insights into where the model excelled and where it needed refinement, particularly in handling
class imbalances, such as the underrepresentation of the "Distribution" topic.


![Confusion Matrix for Topic Detection Model Performance](https://github.com/jaelynnkim-data/nlp-trend-analysis-tool/blob/main/Images%20for%20README/Confusion%20Matrix%20for%20Topic%20Detection%20Model%20Performance.png)


The initial sentiment analysis model, based on ProsusAI/finbert, demonstrated significant
weaknesses, particularly in distinguishing between positive and neutral sentiments, with an
accuracy score of only 0.2769 on a hand-labeled dataset. The model’s struggle to interpret
sentences containing mixed sentiments necessitated a reconsideration of the approach. After
testing both RoBERTa and FinancialBERT models, RoBERTa was selected for further enhancement because, despite its initial lower accuracy (0.6308), it significantly benefited from
the integration of a sentence compression model. The accuracy of the RoBERTa model improved
to 0.6846 when combined with sentence compression, highlighting the impact of this additional
step in refining sentiment classification. 


In contrast, the FinancialBERT model initially had a higher accuracy (0.6385) compared
to RoBERTa alone. However, when coupled with the sentence compression model,
FinancialBERT’s accuracy did not see the same level of improvement as RoBERTa, remaining
at 0.6385. This outcome underscored the unique synergy between the RoBERTa model and
sentence compression in handling the nuanced language found in earnings call transcripts,
making RoBERTa with sentence compression the preferred choice for this task.


![Accuracy Comparison of Sentiment Analysis Models](https://github.com/jaelynnkim-data/nlp-trend-analysis-tool/blob/main/Images%20for%20README/Accuracy%20Comparison%20of%20Sentiment%20Analysis%20Models.png)


It is important to note that while the typical accuracy score for sentiment analysis models
applied to more predictable text formats, such as product reviews or social media posts, typically
ranges between 70% and 85%, the average accuracy for sentiment analysis on more unpredictable and complex text materials like earnings call transcripts or video transcriptions is
generally lower, averaging between 60% and 75%. Given this context, the accuracy achieved by
the RoBERTa model with sentence compression falls within the expected range for this type of
challenging data, further validating the effectiveness of the approach in this specific application.
Overall, the combination of model performance evaluations, supported by visualizations,
provided a clear and actionable understanding of industry trends, risks, and opportunities,
enabling informed investment decisions. Visualizations such as a confusion matrix for topic
detection and a bar chart showing sentiment analysis accuracy improvement were instrumental in
illustrating these findings.



## User Interface 

Being able to predict trend keywords can be very tricky especially when considering how "the next big" trend term of the upcoming future is often a word that may not have existed before, such as a new technology name or a term that is consisted of multiple words that have their own meaning independently but mean something entirely new when placed next to one another (i.e. cloud database, drop shipping, green energy). 

The resulting trend keyword set was able to capture very important information, but due to the fact that it came along with vague keywords that provides fuzzy information when only viewed as a keyword, it was crucial to develop a human-friendly user interface that is simple, effective, and most importantly, makes the most meaningful keywords immediately eye-catching. 

The resulting design of the user interface is as the below image, and this short walkthrough aims to show how intuitive the design of the NLP system and user interface was intended to be. 

Below is a screenshot of the user interface when the user selects "Tyson" and the "2019", "2020" buttons for the keyword extraction filter criteria, and then selects one of the keyword boxes to see the reason behind why the keyword was selected as potentially important for understanding Tyson during this time period.

Before the user selects any of the keyword boxes, the initial screen after selecting the company and year filter values shows the summary of what the company/companies reported as having happened to them during each year in the selected time frame in the following format:

(Summary Content)
  - (Company Name), YYYY

Once the user clicks on a keyword that peaks their interest the most, all the sentences that discussed the keyword appear on the right hand side to show the story/context behind why the keyword gained attention, by which company and what year the sentence is from, and below this there is a sentiment score of whether the keyword iself was overall negative or positive, with 0-40 being negative, 40-60 being neutral and 70-100 being positive. 

In this use case for the screenshot, the selected keyword box was "elections present uncertainty," and the results for this keyword is displayed on the right hand side of the box buttons. 

Below the first section of the user interface, we see the sentiment and word frequency graphs that allows the user to visually study the market outlook based on the topics that they are most interested in, with Tilia having hand-selected the 18 categories that they consistently take into consideration when making investment decisions. 

Underneath the graphs are the list of summaries of the selected topics during each year within the selected timeframe. For example, if the user selects "Distribution" and "2023", it will show the summary of all conversations that involved the topic each during Q1, Q2, Q3 and Q4 of 2023 in the following format:

(Selected Topic) YYYY Qn: (Summary Content)


![UI Image 1](https://github.com/jaelynnkim-data/nlp-trend-analysis-tool/blob/main/Images%20for%20README/UI%20Image%201.png)
![UI Image 2](https://github.com/jaelynnkim-data/nlp-trend-analysis-tool/blob/main/Images%20for%20README/UI%20Image%202.png)


The analysis conducted through the modeling framework revealed several critical insights. These findings provide a comprehensive understanding of the underlying themes present in the earnings call transcripts. One of the most significant insights is the heightened sensitivity to transmittable diseases within the meat production industry, particularly evident in the case of Tyson Foods. Keywords such as "African Swine Fever" and "clinics" frequently appear in Tyson’s transcripts, spanning from the year of 2019 to 2021, highlighting the company’s acute awareness and proactive measures related to disease outbreaks. This focus suggests that Tyson, and likely other major meat producers, are highly vigilant about the impact of transmittable diseases on their operations, directly influencing their supply chain management, employee health initiatives, and overall business continuity.

![UI Image 2](https://github.com/jaelynnkim-data/nlp-trend-analysis-tool/blob/main/Images%20for%20README/UI%20Image%202.png)

In contrast, while African Swine Fever brought significant challenges and losses to Tyson, in the second quarter of 2020 Sysco reported that they were fortunate to avoid any impact from the same outbreak. This difference in outcomes between Tyson and Sysco provides valuable insights for Tilia, suggesting an opportunity to investigate the differences in their distribution and supply systems. By understanding these distinctions, Tilia can develop strategies to help their portfolio companies become more resilient to unexpected external factors such as viruses or bacteria.

![UI Image 2](https://github.com/jaelynnkim-data/nlp-trend-analysis-tool/blob/main/Images%20for%20README/UI%20Image%202.png)

The analysis also highlighted McCormick’s focus on innovating packaging technologies. Notable detections of technology keywords from the first quarter of year of 2023 include newly patented packaging technologies such as "Snap Tight lids" and "color nitrogen-flushed bottles." These innovations demonstrate McCormick's commitment to enhancing product preservation and consumer convenience, which not only differentiates their products in the market but also indicates a strategic focus on sustainability and quality control. Understanding McCormick’s emphasis on packaging technology can inform investment decisions related to long-term investments in companies that prioritize innovation and sustainability.

![UI Image 2](https://github.com/jaelynnkim-data/nlp-trend-analysis-tool/blob/main/Images%20for%20README/UI%20Image%202.png)

Starbucks, on the other hand, places significant emphasis on differentiating their customer experience, which is crucial in the highly saturated coffee market. The transcripts frequently mentioned keywords related to "rewards membership" and "menu innovation," indicating that Starbucks is focusing on enhancing customer loyalty and continuously evolving its offerings to stand out in a crowded field. This insight into Starbucks' strategic priorities highlights the importance of customer engagement and innovation in maintaining a competitive edge.

![UI Image 2](https://github.com/jaelynnkim-data/nlp-trend-analysis-tool/blob/main/Images%20for%20README/UI%20Image%202.png)

Another significant trend observed in the analysis is the shifting focus from "Distribution" to "Supply Chain" starting from the late months of 2020. Historically, mentions of these two topics followed a similar pattern, but during these months, "Supply Chain" became more dominant. This shift likely reflects the broader industry response to the challenges posed by the COVID-19 pandemic, where companies had to pivot from traditional distribution concerns to addressing more complex supply chain disruptions. Interestingly, while the sentiment scores for both topics historically aligned, there was a very brief period in late 2020 when the "Distribution" topic was associated with a significantly more negative sentiment compared to "Supply Chain." This divergence suggests that during the height of the pandemic, there was a concerted effort to view supply chain challenges as opportunities for innovation and resilience, while distribution challenges were seen in a more negative light.

![UI Image 2](https://github.com/jaelynnkim-data/nlp-trend-analysis-tool/blob/main/Images%20for%20README/UI%20Image%202.png)
![UI Image 2](https://github.com/jaelynnkim-data/nlp-trend-analysis-tool/blob/main/Images%20for%20README/UI%20Image%202.png)

However, by 2023, the frequency patterns of "Distribution" and "Supply Chain" topics returned to being almost identical, indicating a normalization of industry focus as companies adapted to the new post-pandemic realities. The alignment of sentiment patterns also returned, suggesting that the industry has regained equilibrium in its approach to managing both distribution and supply chain issues. These findings provide valuable insights that can inform investment strategies. The identification of disease sensitivity in the meat industry, McCormick's focus on packaging innovations, and Starbucks’ emphasis on customer experience and loyalty programs are critical factors that investors should consider. Additionally, the contrast between Tyson and Sysco's experiences with African Swine Fever offers a unique opportunity to explore how different supply chain strategies can mitigate the impact of unforeseen external factors.

In conclusion, the insights derived from the analysis underscore the importance of monitoring external risks, industry innovations, and shifting market dynamics. By leveraging these findings, stakeholders can make more informed decisions that align with emerging trends and potential risks. This proactive approach to understanding market dynamics not only supports better investment decisions but also enhances the ability to respond swiftly to unforeseen challenges.
