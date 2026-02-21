import asyncio
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        res = await client.post("http://localhost:8080/diagnose", json={"symptoms": "головная боль"})
        print(res.text)

asyncio.run(main())
