# RAG Chunking Strategies

import os

from huggingface_hub import InferenceClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from dotenv import load_dotenv


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


# 1) Character Text Splitter: splits text into chunks based on a specified character limit. This is useful for ensuring that each chunk does not exceed a certain size, which can be important for processing large documents.
# default separator is a newline character, but it can be customized to any character or string.
# we can give a chunk size and a chunk overlap to control how much of the text is included in each chunk.



document = """
Google LLC is an American multinational technology corporation focused on information technology, online advertising, search engine technology, email, cloud computing, software, quantum computing, e-commerce, consumer electronics, and artificial intelligence (AI).[9] It has been referred to as "the most powerful company in the world" by the BBC,[10] and is one of the world's most valuable brands.[11][12][13] Google's parent company Alphabet Inc. has been described as a Big Tech company.
Google was founded in 1998 by American computer scientists Larry Page and Sergey Brin. Together, they own about 14% of its publicly listed shares and control 56% of its stockholder voting power through super-voting stock. The company went public via an initial public offering (IPO) in 2004. In 2015, Google was reorganized as a wholly owned subsidiary of Alphabet Inc. Google is Alphabet's largest subsidiary and is a holding company for Alphabet's internet properties and interests. Sundar Pichai was appointed CEO of Google in 2015, replacing Larry Page, who became the CEO of Alphabet. In 2019, Pichai also became the CEO of Alphabet.[14]

After the success of its original service, Google Search (often known simply as "Google"), the company has rapidly grown to offer a multitude of products and services. These products address a wide range of use cases, including email (Gmail), navigation and mapping (Waze, Maps, and Earth), cloud computing (Cloud), web navigation (Chrome), video sharing (YouTube), productivity (Workspace), operating systems (Android and ChromeOS), cloud storage (Drive), language translation (Translate), photo storage (Photos), videotelephony (Meet), smart home (Nest), smartphones (Pixel), wearable technology (Pixel Watch and Fitbit), music streaming (YouTube Music), video on demand (YouTube TV), AI (Google Assistant and Gemini), machine learning APIs (TensorFlow), AI chips (TPU), and more. Many of these products and services are dominant in their respective industries, as is Google Search. Discontinued Google products include gaming (Stadia),[15] Glass, Google+, Reader, Play Music, Nexus, Hangouts, and Inbox by Gmail.[16][17] Google's other ventures outside of internet services and consumer electronics include quantum computing (Willow, Google Quantum AI), self-driving cars (Waymo), and transformer models (Google DeepMind).[18]

Google Search and YouTube are the two most-visited websites worldwide, followed by Facebook, Instagram, and ChatGPT. Google is the largest provider of search engines, mapping and navigation applications, email services, office suites, online video platforms, photo and cloud storage, mobile operating systems, web browsers, machine learning frameworks, and AI virtual assistants in the world as measured by market share.[19] Google was ranked the second most valuable brand by Forbes as of January 2022,[20] and fourth by Interbrand as of February 2022.[21] The company has received criticism involving issues such as privacy concerns, tax avoidance, censorship, search neutrality, antitrust, and abuse of its monopoly position.[22]
Google began in January 1996 as a research project by Larry Page and Sergey Brin[23][24][25] with additional contribution from Scott Hassan.[26][27] Page and Brin published their research, along with Rajeev Motwani and Terry Winograd, as the PageRank algorithm[28] and Stanford owned the patent.[29] PageRank determines a website's relevance by the pages that linked back to the original site.[30][31][28] Page told his ideas to Hassan, who began writing the code to implement Page's ideas.[26] Page and Brin would also use their friend Susan Wojcicki's garage as their office when the search engine was set up in 1998.[32]

Eventually, they changed the name to Google; the name of the search engine was a misspelling of the word googol, a very large number written 10100 (1 followed by 100 zeros), picked to signify that the search engine was intended to search a large number of websites.[30]

Google was initially funded by an August 1998 investment of $100,000 from Andy Bechtolsheim,[23] co-founder of Sun Microsystems. This initial investment served as a motivation to incorporate the company to be able to use the funds.[33][34] Page and Brin initially approached David Cheriton for advice because he had a nearby office in Stanford, and they knew he had startup experience, having recently sold Granite Systems, the company he co-founded with Andy Bechtolsheim, to Cisco for $220 million. Cheriton arranged a meeting at his home in Palo Alto with Page, Brin, and Bechtolsheim. After a brief meeting which included a demo of the website Cheriton and Bechtolsheim agreed to fund Page and Brin's company approximately $200,000 investment each.[35][36]
Google received money from two other angel investors in 1998, including Amazon founder Jeff Bezos, and entrepreneur Ram Shriram.[38] Page and Brin had first approached Shriram, who was a venture capitalist, for funding and counsel, and Shriram invested $250,000 in Google in February 1998. Shriram knew Bezos because Amazon had acquired Junglee, at which Shriram was the president. It was Shriram who told Bezos about Google. Bezos asked Shriram to meet Google's founders and they met six months after Shriram had made his investment when Bezos and his wife were on a vacation trip to the Bay Area. Google's initial funding round had already formally closed but Bezos' status as CEO of Amazon was enough to persuade Page and Brin to extend the round and accept his investment.[39][40] Between these initial investors, friends, and family, Google raised approximately $1,000,000, to fund their original shop in Menlo Park, California.[41] Craig Silverstein, a fellow PhD student at Stanford, was hired as the first employee.[25][42][43]

After some additional small investments through the end of 1998 to early 1999,[38] a new $25 million round of funding was announced on June 7, 1999,[44] with major investors including the venture capital firms Kleiner Perkins and Sequoia Capital.[34] Both firms were initially hesitant about investing jointly in Google, as each wanted to retain a larger percentage of control over the company to themselves. Page and Brin insisted on taking investments from both. Both venture companies finally agreed to investing jointly $12.5 million each due to their belief in Google's great potential and through the mediation of earlier angel investors Ron Conway and Shriram who had contacts in the venture companies.[45]
In March 1999, the company moved its offices to Palo Alto, California,[47] which is home to several prominent Silicon Valley technology start-ups.[48] The next year, Google began selling advertisements associated with search keywords against Page and Brin's initial opposition toward an advertising-funded search engine.[49][25] To maintain an uncluttered page design, advertisements were solely text-based.[50] In June 2000, it was announced that Google would become the default search engine provider for Yahoo!, one of the most popular websites at the time, replacing Inktomi.[51][52]

In 2001, Google's investors felt the need to have a strong internal management, and they agreed to hire Eric Schmidt as the chairman and CEO of Google.[41] Schmidt was proposed by John Doerr from Kleiner Perkins. He had been trying to find a CEO that Page and Brin would accept for several months, but they rejected several candidates because they wanted to retain control over the company. Michael Moritz from Sequoia Capital at one point even menaced requesting Google to immediately pay back Sequoia's $12.5m investment if they did not fulfill their promise to hire a chief executive officer, which had been made verbally during investment negotiations. Schmidt was not initially enthusiastic about joining Google either, as the company's full potential had not yet been widely recognized at the time, and as he was occupied with his responsibilities at Novell where he was CEO. As part of him joining, Schmidt agreed to buy $1 million of Google preferred stocks as a way to show his commitment and to provide funds Google needed.[53]

In 2003, after outgrowing two other locations, the company leased an office complex from Silicon Graphics, at 1600 Amphitheatre Parkway in Mountain View, California.[54] The complex became known as the Googleplex a play on the word googolplex. Three years later, Google bought the property from SGI for $319 million.[55] By that time, the name "Google" had found its way into everyday language, causing the verb "google" to be added to the Merriam-Webster Collegiate Dictionary and the Oxford English Dictionary, denoted as: "to use the Google search engine to obtain information on the Internet".[56][57]
On August 19, 2004, Google became a public company via an initial public offering.[58] The company opened on the NASDAQ National Market under the five-letter ticker symbol GOOGL with an offering of 19,605,052 shares at a price of $85 per share.[59][60] Shares were sold in an online auction format using a system built by Morgan Stanley and Credit Suisse, underwriters for the deal.[61][62] The sale of $1.67 billion gave Google a market capitalization of more than $23 billion.[63]

The company's code of conduct highlighted the motto "Don't be evil". They stated, "We believe strongly that in the long term, we will be better served—as shareholders and in all other ways—by a company that does good things for the world even if we forgo some short term gains."[64]

Google has a dual-class stock structure in which each Class B share gets ten votes compared to each Class A share getting one. Page said in the prospectus that Google has "a dual-class structure that is biased toward stability and independence and that requires investors to bet on the team, especially Sergey and me."[65]
Google's advertising revenue was challenged in 2006 when some advertisers declined to purchase display ads. The company leadership was split on how to handle trade-offs between privacy and advertising value. DoubleClick was known in the industry for the value its cookies brought to the advertising market.[70] In 2007, Google agreed to buy DoubleClick for $3.1 billion and completed the acquisition on March 11, 2008.[70][71][72] After the acquisition, Google was convinced to support advertisers tracking preferences.[70]

Google built 11 data centers around the world with several thousand servers in each. These data centers allowed Google to handle the ever-changing workload more efficiently.[41]

In May 2012, Google acquired Motorola Mobility for $12.5 billion.[73][74][75] This purchase was made in part to help Google gain Motorola's considerable patent portfolio on mobile phones and wireless technologies, to help protect Google in its ongoing patent disputes with other companies,[76] mainly Apple and Microsoft,[77] and to allow it to continue to freely offer Android.[78]
In June 2013, Google acquired Waze for $966 million.[79] While Waze would remain an independent entity, its social features, such as its crowdsourced location platform, were reportedly valuable integrations between Waze and Google Maps, Google's own mapping service.[80]

Google announced the launch of a new company, called Calico, on September 19, 2013, to be led by Apple Inc. chairman Arthur Levinson. In the official public statement, Page explained that the "health and well-being" company would focus on "the challenge of ageing and associated diseases".[81]

On January 26, 2014, Google announced it had agreed to acquire DeepMind Technologies, a privately held AI company from London.[82] Technology news website Recode reported that the company was purchased for $400 million, yet the source of the information was not disclosed. A Google spokesperson declined to comment on the price.[83][84] In 2016, DeepMind's AlphaGo became the first computer program to defeat a top human pro at the game of Go.[85]

Google was ranked second to behind Apple as the most valuable brand in the world from 2013-2016
On August 10, 2015, Google announced plans to reorganize its various interests as a conglomerate named Alphabet Inc. Google became Alphabet's largest subsidiary and the umbrella company for Alphabet's Internet interests. Upon completion of the restructuring, Sundar Pichai became CEO of Google, replacing Page, who became CEO of Alphabet.
Between 2018 and 2019, tensions between the company's leadership and its workers escalated as staff protested company decisions on internal sexual harassment, Dragonfly, a censored Chinese search engine, and Project Maven, a military drone artificial intelligence, which had been seen as areas of revenue growth for the company.[93][94] On October 25, 2018, The New York Times published the exposé, "How Google Protected Andy Rubin, the 'Father of Android'". The company subsequently announced that "48 employees have been fired over the last two years" for sexual misconduct.[95] On November 1, 2018, more than 20,000 Google employees and contractors staged a global walk-out to protest the company's handling of sexual harassment complaints.[96][97] CEO Sundar Pichai was reported to be in support of the protests.[98] Later in 2019, some workers accused the company of retaliating against internal activists.[94]

On March 19, 2019, Google announced that it would enter the video game market, launching a cloud gaming platform called Google Stadia.[99] In March 2021, Google reportedly paid $20 million for Ubisoft ports on Google Stadia.[100] Google spent "tens of millions of dollars" on getting major publishers such as Ubisoft and Take-Two to bring some of their biggest games to Stadia.[101]

On June 3, 2019, the U.S. Department of Justice reported that it would investigate Google for antitrust violations.[102] This led to the filing of an antitrust lawsuit in October 2020, on the grounds the company had abused a monopoly position in the search and search advertising markets.[103] In December 2019, former PayPal chief operating officer Bill Ready became Google's new commerce chief. Ready's role will not be directly involved with Google Pay.[104]

In April 2020, due to the COVID-19 pandemic, Google announced several cost-cutting measures. These measures included slowing down hiring for the remainder of 2020, except for a small number of strategic areas, recalibrating the focus and pace of investments in areas like data centers and machines, and non-business essential marketing and travel.[105] Most employees were also working from home due to the COVID-19 pandemic and the success of it even led to Google announcing that they would be permanently converting some of their jobs to work from home.[106]

"""
chunk_size = 200
chunk_overlap = 40

def character_text_splitter(document, chunk_size=200, chunk_overlap=40):
    """
    Splits the document into smaller chunks based on a specified character limit.

    Args:
        document (str): The document to be split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[str]: A list of chunks.
    """
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(document)

    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n")

# ============================================================================================================
# RECURSSIVE CHARACTER TEXT SPLITTER: splits text into chunks based on a specified character limit, but it does so recursively. This means that if a chunk exceeds the specified limit, it will be further split into smaller chunks until all chunks are within the limit. This is useful for ensuring that each chunk does not exceed a certain size, which can be important for processing large documents.
# ============================================================================================================

def recursive_character_text_splitter(document, chunk_size=200, chunk_overlap=40):
    recursive_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""], chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks_recursive = recursive_splitter.split_text(document)

    print(f"Number of recursive chunks: {len(chunks_recursive)}")
    for i, chunk in enumerate(chunks_recursive):
        print(f"Recursive Chunk {i+1}:\n{chunk}\n")

# ===========================================================================================================
# SEMANTIC CHUNKING STRATEGY: splits text into chunks based on semantic meaning. This means that the text is split into chunks based on the meaning of the text, rather than just the number of characters. This is useful for ensuring that each chunk contains a complete thought or idea, which can be important for processing large documents.
# ===========================================================================================================

def semantic_chunking(document, chunk_size=200, chunk_overlap=40):
    """
    Splits the document into smaller chunks based on semantic meaning.

    Args:
        document (str): The document to be split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[str]: A list of chunks.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70
    )

    chunks_semantic = semantic_splitter.split_text(document)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    semantic_splitter = SemanticChunker(
        embeddings = embeddings,
        breakpoint_threshold_type = "percentile",
        breakpoint_threshold_amount = 70
    )

    chunks_semantic = semantic_splitter.split_text(document)

    print(f"Number of semantic chunks: {len(chunks_semantic)}")
    for i, chunk in enumerate(chunks_semantic):
        print(f"Semantic Chunk {i+1}:\n{chunk}\n")

def agentic_chunking(document, chunk_size=200, chunk_overlap=40):
    """
    Splits the document into smaller chunks based on semantic meaning using an agentic approach.

    Args:
        document (str): The document to be split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[str]: A list of chunks.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    prompt = f"""

    You are an expert in text chunking. Split this this into logical chunks.
    Rules:
    - Each chunk should be around 200 characters or less.
    - Split at natural topic boundaries.
    - Keep related information together.
    - Put "<<<Split>>>" between each chunk.

    Text: {document}

    Return the text with "<<<Split>>>" between each chunk. Do not add any additional text or explanation.
    """
    print("Asking AI to chunk the text...")

    client = InferenceClient(
        model="Qwen/Qwen3-8B",
        token=HF_TOKEN
    )
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
    )
    marked_text = response.choices[0].message.content

    chunks = marked_text.split("<<<Split>>>")
    clean_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    print(f"Number of agentic chunks: {len(clean_chunks)}")
    for i, chunk in enumerate(clean_chunks):
        print(f"Agentic Chunk {i+1}:\n{chunk}\n")    

def main():

    # print("Character Text Splitter:")
    # character_text_splitter(document, chunk_size, chunk_overlap)

    # print("\nRecursive Character Text Splitter:")
    # recursive_character_text_splitter(document, chunk_size, chunk_overlap)

    # print("\nSemantic Chunking Strategy:")
    # semantic_chunking(document, chunk_size, chunk_overlap)

    print("\nAgentic Chunking Strategy:")
    agentic_chunking(document, chunk_size, chunk_overlap)

if __name__ == "__main__":
    main()
