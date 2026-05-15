"""
我们已经爬取了 FAQ 文档，接下来就需要对收集到的文档进行统一处理，内容包括：

文本清洗（去除 HTML 标签、无关字符）
分段切分（按规则或语义将文档拆分成小片段，便于检索）
元数据标注（来源、时间、业务类别等）。
代码如下：
"""

import json
from bs4 import BeautifulSoup
from langchain_core.documents import Document

def parse_faq_html(file_path):
    """
    解析FAQ HTML文件，提取问题和答案信息并封装为Document对象列表。

    参数:
        file_path (str): FAQ HTML文件的路径。

    返回:
        list: 包含Document对象的列表，每个对象的metadata包含分类、问题、答案和来源信息。
    """
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    current_category = None

    # 遍历所有<ul>标签，解析其中的<li>元素
    for ul in soup.find_all("ul"):
        for li in ul.find_all("li", recursive=False):
            h1 = li.find("h1")
            if h1:  # 分类标题
                current_category = h1.get_text(strip=True)
                continue

            dl = li.find("dl")
            if dl:
                # 去掉 Q：
                question_raw = dl.find("dt").get_text(strip=True)
                question = question_raw.lstrip("Q：").strip()
                answer = dl.find("dd").get_text(strip=True)

                docs.append(
                    Document(
                        page_content="",
                        metadata={
                            "source": file_path,
                            "category": current_category,
                            "question": question,
                            "answer": answer
                        }
                    )
                )
    return docs

def save_docs_to_json(docs, output_file):
    """
    将Document对象列表保存为JSON格式文件。

    参数:
        docs (list): 包含Document对象的列表。
        output_file (str): 输出JSON文件的路径。

    返回:
        None
    """
    data = [
        {
            "question": doc.metadata["question"],
            "answer": doc.metadata["answer"],
            "category": doc.metadata["category"],
            "source": doc.metadata["source"]
        }
        for doc in docs
    ]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"FAQ 已保存到 {output_file}")

if __name__ == "__main__":
    faq_docs = parse_faq_html("faq.html")
    for d in faq_docs:
        print(d.metadata)
    save_docs_to_json(faq_docs, "faq.json")

