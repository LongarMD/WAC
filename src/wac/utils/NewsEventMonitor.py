from wac.utils.NewsEvent import NewsEvent
from wac.utils.NewsArticle import NewsArticle

from typing import List, Union

# ===============================================
# Define the News Event Monitor
# ===============================================


class NewsEventMonitor:
    active_events: List[NewsEvent]
    past_events: List[NewsEvent]

    def __init__(self, strategy) -> None:
        self.active_events = []
        self.past_events = []
        self.strategy = strategy

    # ==================================
    # Default Override Methods
    # ==================================

    # ==================================
    # Class Methods
    # ==================================

    def update(self, input: Union[NewsArticle, NewsEvent], device=None):
        """Update the events with the new article"""

        # get the event that matches all of the conditions
        event = self.strategy.find_relevant_event(
            input, self.active_events, device=device
        )
        # update active events
        self.strategy.update_active_events(input, event, self.active_events)

        # remove events that are old
        self.__update_past_events(input.date_time)

    @property
    def events(self):
        """Get all of the events"""
        return self.past_events + self.active_events

    # ==================================
    # Remove Methods
    # ==================================

    def __update_past_events(self, date_time):
        """Update the past events"""
        for event_id in reversed(range(len(self.active_events))):
            event = self.active_events[event_id]
            if self.strategy.sunset_event(event, date_time=date_time):
                # add the event to the past events
                self.past_events.append(event)
                # remove the event from the active events
                del self.active_events[event_id]

    # ==================================
    # Evaluation Methods
    # ==================================

    def assign_cluster_ids_to_articles(self):
        """Assigns the articles associated event ID"""
        # iterate through every articles
        events = self.past_events + self.active_events
        # iterate through each possible language
        for idx, event in enumerate(events):
            event.assign_cluster_id(f"{idx}")
