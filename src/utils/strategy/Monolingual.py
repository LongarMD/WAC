from typing import List

import torch
import numpy as np

from src.utils.NewsEvent import NewsEvent
from src.utils.NewsArticle import NewsArticle

from src.utils.LinearAlgebra import cosine_similarity

# ================================================
# Constants
# ================================================

# one day in seconds
ONE_DAY = 86400

SIM_FIELDS = ["title", "body"]
TIME_METRICS = ["min", "max", "avg"]


# ================================================
# Helper functions
# ================================================


def time_norm(x, nu=0, delta=1):
    return np.exp(-0.5 * ((x - nu) / delta) ** 2)


# ================================================
# Strategy definition
# ================================================


class MonolingualStrategy:
    def __init__(self, rank_th: float = 0.5, time_std: float = 3) -> None:
        self.rank_th = rank_th
        self.time_std = time_std

    def find_relevant_event(
        self, article: NewsArticle, active_events: List[NewsEvent], **kwargs
    ) -> NewsEvent:
        # get the events of the specific language
        lang_events = [e for e in active_events if e.langs[0] == article.lang]

        if len(lang_events) == 0:
            # there are no events of the specific language
            return None

        # calculate the rank of the events
        sims = torch.Tensor([self.__get_rank(article, event) for event in lang_events])
        # get the sorted indices of the highest-ranking events
        sort_index = torch.argsort(sims, descending=True)

        if sims[sort_index[0]] > self.rank_th:
            # get the highest-ranking news event
            event = lang_events[sort_index[0]]
            return event

        # no appropriate event found
        return None

    def update_active_events(
        self, article: NewsArticle, event: NewsEvent, events: List[NewsEvent], **kwargs
    ):
        if event is not None:
            # add update the event
            event.add_article(article)
        else:
            # create a new event cluster
            event = NewsEvent(articles=[article])
            events.append(event)

    def sunset_event(self, event: NewsEvent, date_time: float, **kwargs):
        return self.__get_time_similarity(event, date_time) <= 0.3

    # ==================================
    # Helper Methods
    # ==================================

    def __get_time_prob(self, time1: float, time2: float) -> float:
        diff_in_days = abs(time1 - time2) / ONE_DAY
        return time_norm(diff_in_days, delta=self.time_std)

    def __get_time_similarity(self, event: NewsEvent, date_time: float) -> float:
        time_scales = [
            self.__get_time_prob(date_time, event.get_time(metric=metric))
            for metric in TIME_METRICS
        ]
        return sum(time_scales) / len(TIME_METRICS)

    def __get_content_similarity(self, article: NewsArticle, event: NewsEvent) -> float:
        sim_scales = [
            cosine_similarity(
                article.get_embedding(type=field1),
                event.centroids[field2]["vector"],
            )
            for field1 in SIM_FIELDS
            for field2 in SIM_FIELDS
        ]
        return sum(sim_scales) / (len(SIM_FIELDS) ** 2)

    def __get_rank(self, article: NewsArticle, event: NewsEvent) -> float:
        time_sim = self.__get_time_similarity(event, article.date_time)
        content_sim = self.__get_content_similarity(article, event)
        return time_sim * content_sim
