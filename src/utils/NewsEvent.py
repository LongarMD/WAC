import operator
from functools import reduce

from src.utils.NewsEventBase import NewsEventBase
from src.utils.LinearAlgebra import (
    get_centroid,
    update_centroid,
)


class NewsEvent(NewsEventBase):
    """The class describing the monolingual event instance"""

    def __init__(self, articles=[]):
        super().__init__(articles)

        # initialize the event properties
        self.centroids = {
            "title": {"vector": None, "norm": 0},
            "body": {"vector": None, "norm": 0},
            "content": {"vector": None, "norm": 0},
        }
        self.named_entities = None
        # update the event properties
        self.__init_centroids()

    # ==================================
    # Class Methods
    # ==================================

    def add_article(self, article):
        super().add_article(article)
        # update the event values
        self.__update_centroids()

    def add_articles(self, articles):
        # append the articles
        for article in articles:
            self.add_article(article)

    # ==================================
    # Centroid Methods
    # ==================================

    def __init_centroids(self):
        for key in self.centroids.keys():
            self.centroids[key]["vector"], self.centroids[key]["norm"] = get_centroid(
                self.get_article_embeddings(type=key).squeeze(0)
            )

    def __update_centroids(self):
        if len(self.articles) == 1:
            for key in self.centroids.keys():
                (
                    self.centroids[key]["vector"],
                    self.centroids[key]["norm"],
                ) = get_centroid(self.get_article_embeddings(type=key).squeeze(0))
        else:
            # update the cluster centroid
            n_articles = len(self.articles) - 1
            for key in self.centroids.keys():
                (
                    self.centroids[key]["vector"],
                    self.centroids[key]["norm"],
                ) = update_centroid(
                    self.centroids[key]["vector"],
                    self.centroids[key]["norm"],
                    n_articles,
                    self.articles[-1].get_embedding(type=key),
                )
