import streamlit as st



def example():
    st.code("""Create a final answer and reason out three Suggested Questions to the given questions using the provided document excerpts (given in no particular order) as sources. ALWAYS include a "SOURCES" section in your answer citing only the minimal set of sources needed to answer the question. If you are unable to answer the question, simply state that you do not have enough information to answer the question and leave the SOURCES section empty. Use only the provided documents and do not attempt to fabricate an answer. You should always analyze and reason out the three related associations behind the answer to the question. Always include a "Suggested Questions" section in your answer

---------

QUESTION: What is Belt and Road Initiative?
=========
Content: The Belt and Road Initiative (BRI, or B&R[1]), known within China as the One Belt One Road (Chinese: 一带一路; pinyin: Yīdài Yīlù) or OBOR/1B1R for short,[2] is a global infrastructure development strategy adopted by the Chinese government in 2013 to invest in more than 150 countries and international organizations.
SOURCES: 1-32
Content: According to London-based consultants Centre for Economics and Business Research, BRI is likely to increase the world GDP by $7.1 trillion per annum by 2040, and that benefits will be "widespread" as improved infrastructure reduces "frictions that hold back world trade". CEBR also concludes that the project will be likely to attract further countries to join, if the global infrastructure initiative progresses and gains momentum.
SOURCES: 1-33
Content: Supporters praise the BRI for its potential to boost the global GDP, particularly in developing countries. However, there has also been criticism over human rights violations and environmental impact, as well as concerns of debt-trap diplomacy resulting in neocolonialism and economic imperialism. These differing perspectives are the subject of active debate.
SOURCES: 1-30
=========
FINAL ANSWER: The Belt and Road Initiative (BRI, or B&R[1]), known within China as the One Belt One Road (Chinese: 一带一路; pinyin: Yīdài Yīlù) or OBOR/1B1R for short,[2] is a global infrastructure development strategy adopted by the Chinese government in 2013 to invest in more than 150 countries and international organizations.
SOURCES: 1-32
Suggested Questions: 
- What are the outcomes of the Belt and Road Initiative?
- What are the challenges of the Belt and Road Initiative?
- Which country initiated the Belt and Road Initiative?
---------

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:""", language='python')
