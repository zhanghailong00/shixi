"""
收集美团客服相关知识文档，例如：

业务手册（退款规则、订单处理流程）
常见问题 FAQ
内部客服知识库
实时更新的运营公告
以美团外卖常见问题为例，文档地址：https://waimai.meituan.com/help/faq，我们通过playwright 工具爬虫获取页面数据并写入本地 txt 文件中。

chromium下载非常慢，推荐换成chrome浏览器，并指定路径：
"""

from playwright.sync_api import sync_playwright

def collect_faq(url):
    """
    收集指定URL页面中的FAQ内容

    参数:
        url (str): 目标网页URL地址

    返回:
        str: 提取的FAQ内容html代码
    """
    # 启动Playwright浏览器自动化工具
    with sync_playwright() as p:
        # 启动Chromium浏览器，设置为非无头模式并指定中文语言
        browser = p.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            headless=False,
            args=['--lang=zh-CN']  # 浏览器语言
        )
        # 创建新页面，配置中文环境
        page = browser.new_page(
            locale='zh-CN',  # 页面 locale
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0"
            ),
            extra_http_headers={
                "Accept-Language": "zh-CN,zh;q=0.9"
            }
        )
        # 访问目标URL并等待页面加载完成
        page.goto(url, timeout=30_000)
        page.wait_for_load_state("networkidle")

        # 提取FAQ列表区域的html代码
        raw_text = page.locator("#faq-list").first.inner_html()
        browser.close()
        return raw_text


def save_faq(cleaned_text: str, output_file: str):
    """
    将FAQ内容保存到指定文件

    参数:
        cleaned_text (str): 要保存的FAQ内容
        output_file (str): 输出文件路径
    """
    # 写入文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"FAQ 已保存到 {output_file}")


if __name__ == "__main__":
    cleaned_text = collect_faq(url="https://waimai.meituan.com/help/faq")
    output_file = "faq.html"
    save_faq(cleaned_text, output_file)