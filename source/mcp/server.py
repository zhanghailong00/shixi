import json
import os
import httpx
import dotenv
from loguru import logger
from mcp.server.fastmcp import FastMCP

dotenv.load_dotenv()

# =========================
# 创建 MCP Server（SSE模式）
# =========================
mcp = FastMCP(
    name="WeatherServerSSE",
    host="0.0.0.0",
    port=8000
)


# =========================
# 工具：天气查询
# =========================
@mcp.tool()
def get_weather(city: str) -> str:
    """
    查询指定城市的即时天气信息（OpenWeather API）

    参数:
        city: 城市英文名，如 Beijing / Shanghai

    返回:
        JSON字符串（天气数据）
    """

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": os.getenv("OPENWEATHER_API_KEY"),
        "units": "metric",
        "lang": "zh_cn"
    }

    # 调用外部API
    resp = httpx.get(url, params=params, timeout=10)
    data = resp.json()

    logger.info(f"[MCP] {city} 天气查询完成")

    return json.dumps(data, ensure_ascii=False)


# =========================
# 启动 MCP SSE Server
# =========================
if __name__ == "__main__":
    logger.info("🚀 MCP Weather Server 启动中...")
    logger.info("📡 SSE地址: http://0.0.0.0:8000/sse")

    # SSE模式启动（MCP标准传输方式）
    mcp.run(transport="sse")