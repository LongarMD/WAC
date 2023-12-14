from typing import List

import torch
import numpy as np

from src.utils.NewsEvent import NewsEvent
from src.utils.Wasserstein import Wasserstein

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


class EventStrategy:
    def __init__(
        self,
        rank_th: float = 0.7,
        time_std: float = 3,
        pre_sim_n: int = 1000,
        w_reg: float = 0.1,
        w_nit: int = 10,
        device="cpu",
    ) -> None:
        self.sim_th = rank_th
        self.time_th = time_std
        self.pre_sim_n = pre_sim_n
        self.wasserstein = Wasserstein(reg=w_reg, nit=w_nit, device=device)

    def find_relevant_event(
        self, target_event: NewsEvent, active_events: List[NewsEvent], **kwargs
    ) -> NewsEvent:
        # get the events of the specific language

        if len(active_events) == 0:
            # there are no events of the specific language
            return None

        # prior selection of events based on the centriod similarity
        pre_sim = torch.Tensor(
            [
                self.__get_rank(target_event, event, sim_type="cosine")
                for event in active_events
            ]
        )
        pre_sim_idx = torch.argsort(pre_sim, descending=True)
        sim_events = [active_events[idx] for idx in pre_sim_idx[: self.pre_sim_n]]

        # calculate the rank of the events
        sims = torch.Tensor(
            [
                self.__get_rank(target_event, event, sim_type="wasserstein")
                for event in sim_events
            ]
        )
        # get the sorted indices of the highest-ranking events
        sort_index = torch.argsort(sims, descending=True)

        if sims[sort_index[0]] > self.sim_th:
            # get the highest-ranking news event
            event = sim_events[sort_index[0]]
            return event

        # no appropriate event found
        return None

    def update_active_events(
        self,
        curr_event: NewsEvent,
        rank_event: NewsEvent,
        active_events: List[NewsEvent],
        **kwargs,
    ):
        if rank_event is not None:
            # add update the event
            rank_event.add_articles(curr_event.articles)
        else:
            # add the event cluster to the active list
            active_events.append(curr_event)

    def sunset_event(self, event: NewsEvent, date_time: dict, **kwargs):
        return self.__get_time_similarity(event, date_time) <= 0.3

    # ==================================
    # Helper Methods
    # ==================================

    def __get_time_prob(self, time1: float, time2: float) -> float:
        diff_in_days = abs(time1 - time2) / ONE_DAY
        return time_norm(diff_in_days, delta=self.time_th)

    def __get_time_similarity(self, event: NewsEvent, date_time: dict) -> float:
        time_scales = torch.tensor(
            [
                [
                    self.__get_time_prob(
                        date_time[metric1], event.get_time(metric=metric2)
                    )
                    for metric2 in TIME_METRICS
                ]
                for metric1 in TIME_METRICS
            ]
        )
        P = torch.sum(torch.max(time_scales, dim=1)[0]) / len(TIME_METRICS)
        R = torch.sum(torch.max(time_scales, dim=0)[0]) / len(TIME_METRICS)
        F1 = 2 * P * R / (P + R)
        return F1  # sum(time_scales) / len(TIME_METRICS)

    def __get_wasserstein_similarity(
        self, curr_event: NewsEvent, rank_event: NewsEvent
    ) -> float:
        t_size = len(curr_event.articles)
        m_size = len(rank_event.articles)
        # initialize the cost matrix
        C = torch.zeros((1, t_size, m_size))
        # create the cost matrix based on the article embeddings
        for type in SIM_FIELDS:
            t_embed = curr_event.get_article_embeddings(type=type)
            m_embed = rank_event.get_article_embeddings(type=type)
            torch.add(C, self.wasserstein.get_cost_matrix(t_embed, m_embed), out=C)
        torch.div(C, len(SIM_FIELDS), out=C)
        # correct the similarity matrix based on the language
        L = self.__get_lang_correction(curr_event.langs, rank_event.langs)
        torch.sub(C, torch.mul(C, L), alpha=0.01, out=C)

        t_dist = self.wasserstein.get_distributions(torch.ones((1, t_size)))
        m_dist = self.wasserstein.get_distributions(torch.ones((1, m_size)))
        sim, _, _ = self.wasserstein(C, t_dist, m_dist, as_prob=True)
        return sim

    def __get_cosine_similarity(
        self, curr_event: NewsEvent, rank_event: NewsEvent
    ) -> float:
        sim_scales = [
            cosine_similarity(
                curr_event.centroids[field1]["vector"],
                rank_event.centroids[field2]["vector"],
            )
            for field1 in SIM_FIELDS
            for field2 in SIM_FIELDS
        ]
        return sum(sim_scales) / (len(SIM_FIELDS) ** 2)

    def __get_rank(
        self, curr_event: NewsEvent, rank_event: NewsEvent, sim_type: str
    ) -> float:
        time_sim = self.__get_time_similarity(rank_event, curr_event.date_time)

        if sim_type == "wasserstein":
            content_sim = self.__get_wasserstein_similarity(curr_event, rank_event)
        elif sim_type == "cosine":
            content_sim = self.__get_cosine_similarity(curr_event, rank_event)
        else:
            raise NotImplementedError(f"Similarity type {sim_type} not implemented")

        return time_sim * content_sim

    def __get_lang_correction(self, curr_langs, rank_langs):
        lang_corr = torch.tensor(
            [
                [int(curr_langs[i] != rank_langs[j]) for j in range(len(rank_langs))]
                for i in range(len(curr_langs))
            ]
        ).unsqueeze(0)
        return lang_corr
