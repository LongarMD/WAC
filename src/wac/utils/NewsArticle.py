import re
import warnings
import datetime
from typing import Any, Set, List, Tuple, TypedDict

import numpy as np
import torch

# ===============================================
# Helper Functions
# ===============================================

# format the strings
regex_whitespace = re.compile(r"(\s){1,}", re.IGNORECASE)


def format_string(x, regex=regex_whitespace):
    """Format the string"""
    return re.sub(regex, " ", x).strip()


# ===============================================
# Define new Types
# ===============================================


class Article(TypedDict):
    """The class describing the article attributes"""

    id: Any
    title: str
    body: str
    lang: str
    date_time: datetime.datetime
    cluster_id: str


# ===============================================
# Define the News Article
# ===============================================


class NewsArticle:
    """The class describing the article instance"""

    rep_model: Any = None
    ner_model: Any = None

    # format="%Y-%m-%dT%H:%M:%SZ"
    def __init__(self, article: Article) -> None:
        self.id = article["id"]
        self.title = format_string(
            article["title"] if isinstance(article["title"], str) else ""
        )
        self.body = format_string(
            article["body"] if isinstance(article["body"], str) else ""
        )
        self.date_time = article["date_time"].timestamp()
        self.lang = article["lang"]
        self.cluster_id = article.get("cluster_id", None)

        # representation placeholders
        self.__content_embedding = None
        self.__title_embedding = None
        self.__body_embedding = None
        self.__named_entities = None

    # ==================================
    # Default Override Methods
    # ==================================

    def __repr__(self) -> str:
        return (
            f"NewsArticle(\n  "
            f"title={self.title},\n  "
            f"body={self.body[0:1000]},\n"
            f"time={self.date_time},\n  "
            f"lang={self.lang},\n  "
            ")"
        )

    def __eq__(self, article: "NewsArticle") -> bool:
        return (
            self.title == article.title
            and self.body == article.body
            and self.lang == article.lang
            and self.date_time == article.date_time
        )

    def __ne__(self, article: "NewsArticle") -> bool:
        return not self == article

    def __ge__(self, article: "NewsArticle") -> bool:
        return self.date_time >= article.date_time

    def __gt__(self, article: "NewsArticle") -> bool:
        return self.date_time > article.date_time

    def __lt__(self, article: "NewsArticle") -> bool:
        return self.date_time < article.date_time

    def __le__(self, article: "NewsArticle") -> bool:
        return self.date_time <= article.date_time

    # ==================================
    # Class Methods
    # ==================================

    def to_array(self):
        return [
            self.id,
            self.title,
            self.body,
            self.lang,
            self.get_time(),
            self.cluster_id,
        ]

    def get_text(self, fields: List[str] = ["title", "body"]) -> str:
        return " ".join([getattr(self, field) for field in fields])

    def get_embedding(self, type="content") -> torch.Tensor:
        if NewsArticle.rep_model is None:
            raise RuntimeError("Embedding model not initialized")

        if type == "content":
            return self.__get_content_embedding()
        elif type == "title":
            return self.__get_title_embedding()
        elif type == "body":
            return self.__get_body_embedding()
        else:
            raise ValueError(f"Unknown embedding type: {type}")

    def __get_content_embedding(self) -> torch.Tensor:
        """Gets the content embedding
        Returns:
            embedding (torch.Tensor): The article content embedding.
        """

        if torch.is_tensor(self.__content_embedding):
            # embedding is already available
            return self.__content_embedding

        # get the content representation
        text = self.get_text()
        self.__content_embedding = NewsArticle.rep_model(text)[0]
        # return the content embedding
        return self.__content_embedding

    def __get_title_embedding(self) -> torch.Tensor:
        """Gets the article title embedding
        Returns:
            embedding (torch.Tensor): The article title embedding.
        """
        if torch.is_tensor(self.__title_embedding):
            # embedding is already available
            return self.__title_embedding

        # get the content representation
        text = self.get_text(fields=["title"])
        self.__title_embedding = NewsArticle.rep_model(text)[0]
        # return the content embedding
        return self.__title_embedding

    def __get_body_embedding(self) -> torch.Tensor:
        """Gets the article body embedding
        Returns:
            embedding (torch.Tensor): The article body embedding.
        """
        if torch.is_tensor(self.__body_embedding):
            # embedding is already available
            return self.__body_embedding

        # get the content representation
        text = self.get_text(fields=["body"])
        self.__body_embedding = NewsArticle.rep_model(text)[0]
        # return the content embedding
        return self.__body_embedding

    def get_time(self) -> datetime.datetime:
        """Gets the article time in a readable format
        Returns:
            time (datetime.datetime): The time when the article
                was published.
        """
        return datetime.datetime.fromtimestamp(self.date_time)

    def get_named_entities(self) -> List[Tuple[str, str]]:
        if NewsArticle.ner_model is None:
            raise RuntimeError("Named entity recognition model not initialized")

        if isinstance(self.__named_entities, set):
            # named entities are already available
            return self.__named_entities

        # get the named entities
        named_entities = NewsArticle.ner_model(self.get_text())
        self.__named_entities = set([e[0] for e in named_entities])
        # return the named entities
        return self.__named_entities
