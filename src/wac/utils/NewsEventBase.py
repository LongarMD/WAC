import torch

from src.utils.LinearAlgebra import (
    get_min,
    get_max,
    get_avg,
)


class NewsEventBase:
    """The class describing the monolingual event instance"""

    def __init__(self, articles=[]):
        # initialize the event properties
        self.articles = articles
        self.date_time = None

        # update the event properties
        self.__get_date_time()

    # ==================================
    # Default Override Methods
    # ==================================

    def __repr__(self):
        return f"NewsEvent(\n  " f"n_articles={len(self.articles)},\n" ")"

    @property
    def min_time(self):
        return self.get_time(metric="min")

    @property
    def avg_time(self):
        return self.get_time(metric="avg")

    @property
    def max_time(self):
        return self.get_time(metric="max")

    @property
    def cluster_ids(self):
        return [a.cluster_id for a in self.articles] if len(self.articles) > 0 else None

    @property
    def langs(self):
        return [a.lang for a in self.articles] if len(self.articles) > 0 else None

    # ==================================
    # Class Methods
    # ==================================

    def add_article(self, article):
        # append the article
        self.articles.append(article)
        # update the event values
        self.__get_date_time()

    def add_articles(self, articles):
        self.articles.extend(articles)
        # update the event values
        self.__get_date_time()

    def get_article_embeddings(self, type="content"):
        return torch.stack(
            [article.get_embedding(type=type) for article in self.articles]
        ).unsqueeze(0)

    def get_time(self, metric="avg"):
        if len(self.articles) == 0:
            return None
        return self.date_time.get(metric, None)

    def assign_cluster_id(self, cluster_id):
        for article in self.articles:
            article.cluster_id = cluster_id

    # ==================================
    # Initialization Methods
    # ==================================

    def __get_date_time(self):
        if len(self.articles) == 0:
            # there are no articles
            self.date_time = None
            return
        # get the article times
        times = [a.date_time for a in self.articles]
        self.date_time = {
            "min": get_min(times),
            "avg": get_avg(times),
            "max": get_max(times),
        }
