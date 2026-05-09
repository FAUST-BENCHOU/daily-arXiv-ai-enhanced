from pydantic import BaseModel, Field, field_validator


class DigestTheme(BaseModel):
    title: str = Field(description="三个主题之一：对应一篇论文方向的简短标题（中文）")
    blurb: str = Field(description="一句话概括该主题关切的问题或亮点")


class DigestStructured(BaseModel):
    themes: list[DigestTheme] = Field(
        description="恰好三项，对应三篇论文方向，用于邮件摘要与 digest 页顶部展示"
    )
    markdown_digest: str = Field(
        description=(
            "完整 Markdown 正文：从「当日文献池编号一览」起，包含 skill 要求的"
            "三篇论文六创新点、矩阵、参考文献总表、自检等；不要再用列表重复 themes 三条标题。"
        )
    )

    @field_validator("themes")
    @classmethod
    def exactly_three_themes(cls, v: list[DigestTheme]) -> list[DigestTheme]:
        if len(v) != 3:
            raise ValueError("themes 必须为恰好 3 项")
        return v
