import operator
from functools import reduce

from wac.utils.NewsEventBase import NewsEventBase
from wac.utils.LinearAlgebra import (
    get_centroid,
    update_centroid,
)


class NewsEvent(NewsEventBase):
    use_ne: bool = False

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
        if NewsEvent.use_ne:
            self.__init_named_entities()

    # ==================================
    # Class Methods
    # ==================================

    def add_article(self, article):
        super().add_article(article)
        # update the event values
        self.__update_centroids()
        if NewsEvent.use_ne:
            self.__update_named_entities()

    def add_articles(self, articles):
        # append the articles
        super().add_articles(articles)
        self.__init_centroids()
        if NewsEvent.use_ne:
            self.__init_named_entities()

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

    # ==================================
    # Entities Methods
    # ==================================

    def __init_named_entities(self):
        if len(self.articles) == 0:
            # there are no articles
            self.named_entities = set()
            return
        # get the article named entities
        ne = [a.get_named_entities() for a in self.articles]
        self.named_entities = reduce(operator.or_, ne)

    def __update_named_entities(self):
        if len(self.articles) == 0:
            # there are no named entities to extract
            self.named_entities = set()
        elif len(self.articles) == 1:
            # there is only one article to extract named entities from
            self.named_entities = self.articles[0].get_named_entities()
        else:
            # append the latest named entities to the cluster
            ne = self.articles[-1].get_named_entities()
            self.named_entities = self.named_entities | ne
