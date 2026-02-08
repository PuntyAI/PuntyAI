"""Twitter/X message formatter."""

import re
from typing import Optional


class TwitterFormatter:
    """Format content for Twitter/X posts.

    Twitter formatting:
    - 280 character limit per tweet
    - Threads for longer content
    - Hashtags for discoverability
    - Mentions with @
    - Emojis supported
    - No markdown (plain text only)
    """

    MAX_TWEET_LENGTH = 280
    THREAD_MAX_TWEETS = 10

    # Racing hashtags
    HASHTAGS = {
        "general": ["#AusRacing", "#HorseRacing"],
        "melbourne": ["#MelbourneRacing", "#Flemington", "#Caulfield", "#MooneeValley"],
        "sydney": ["#SydneyRacing", "#Randwick", "#Rosehill"],
        "brisbane": ["#BrisbaneRacing", "#Doomben", "#EagleFarm"],
        "featured": ["#MelbourneCup", "#CoxPlate", "#GoldenSlipper"],
    }

    # Emojis (used sparingly per PUNTY_MASTER rules)
    EMOJIS = {
        "horse": "\U0001F40E",
        "racing": "\U0001F3C7",
        "fire": "\U0001F525",
        "money": "\U0001F4B0",
        "trophy": "\U0001F3C6",
        "star": "\u2B50",
        "check": "\u2705",
        "dice": "\U0001F3B2",
        "alert": "\U0001F6A8",
    }

    @classmethod
    def format(
        cls,
        raw_content: str,
        content_type: str = "early_mail",
        venue: Optional[str] = None,
    ) -> str:
        """Format raw content for Twitter/X as a single long-form post.

        X supports long-form posts, so content is returned as one post
        with hashtags appended at the end.

        Args:
            raw_content: The raw content to format
            content_type: Type of content
            venue: Venue name for hashtags

        Returns:
            Twitter-formatted content as a single post
        """
        content = cls._clean_markdown(raw_content)
        hashtags = cls._get_hashtags(venue)
        return f"{content}\n\n{hashtags}"

    @classmethod
    def format_as_thread(
        cls,
        raw_content: str,
        content_type: str = "early_mail",
        venue: Optional[str] = None,
    ) -> list[str]:
        """Format content as a list of tweets for a thread.

        Returns:
            List of individual tweet strings
        """
        content = cls._clean_markdown(raw_content)
        hashtags = cls._get_hashtags(venue)

        return cls._split_into_thread(content, hashtags, content_type, venue)

    # Words that trigger X's hateful-conduct filter, mapped to safe replacements
    PROFANITY_SUBS = [
        (re.compile(r'\bCUNT FACTORY\b', re.IGNORECASE), 'CHAOS FACTORY'),
        (re.compile(r'\bcunts?\b', re.IGNORECASE), 'legends'),
        (re.compile(r'\bfuck(ing|ed|s)?\b', re.IGNORECASE), 'bloody'),
    ]

    @classmethod
    def _clean_markdown(cls, content: str) -> str:
        """Remove markdown formatting and sanitise for X."""
        # Remove headers and any leading section numbers like "### 1) HEADER"
        content = re.sub(r'^#{1,3}\s+\d+\)\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'^#{1,3}\s+', '', content, flags=re.MULTILINE)
        # Remove bold
        content = re.sub(r'\*\*(.+?)\*\*', r'\1', content)
        content = re.sub(r'__(.+?)__', r'\1', content)
        content = re.sub(r'\*(.+?)\*', r'\1', content)
        # Remove italic
        content = re.sub(r'_(.+?)_', r'\1', content)
        # Clean up multiple newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        # Sanitise profanity for X content policy
        for pattern, replacement in cls.PROFANITY_SUBS:
            content = pattern.sub(replacement, content)
        return content.strip()

    @classmethod
    def _get_hashtags(cls, venue: Optional[str] = None) -> str:
        """Get relevant hashtags for venue."""
        tags = list(cls.HASHTAGS["general"])

        if venue:
            venue_lower = venue.lower()
            if any(v in venue_lower for v in ["flemington", "caulfield", "moonee"]):
                tags.extend(cls.HASHTAGS["melbourne"][:2])
            elif any(v in venue_lower for v in ["randwick", "rosehill", "warwick"]):
                tags.extend(cls.HASHTAGS["sydney"][:2])
            elif any(v in venue_lower for v in ["doomben", "eagle"]):
                tags.extend(cls.HASHTAGS["brisbane"][:2])

        return " ".join(tags[:3])  # Max 3 hashtags

    @classmethod
    def _format_single_tweet(
        cls,
        content: str,
        hashtags: str,
        content_type: str,
    ) -> str:
        """Format as single tweet."""
        emoji = cls.EMOJIS.get("racing", "")

        # Trim content if needed
        max_content_len = cls.MAX_TWEET_LENGTH - len(hashtags) - len(emoji) - 4
        if len(content) > max_content_len:
            content = content[:max_content_len - 3] + "..."

        return f"{emoji} {content}\n\n{hashtags}"

    @classmethod
    def _format_thread(
        cls,
        content: str,
        hashtags: str,
        content_type: str,
        venue: Optional[str],
    ) -> str:
        """Format as thread with numbered tweets."""
        tweets = cls._split_into_thread(content, hashtags, content_type, venue)

        # Format as readable thread preview
        result = []
        for i, tweet in enumerate(tweets, 1):
            result.append(f"Tweet {i}:\n{tweet}")

        return "\n\n".join(result)

    @classmethod
    def _split_into_thread(
        cls,
        content: str,
        hashtags: str,
        content_type: str,
        venue: Optional[str],
    ) -> list[str]:
        """Split content into thread of tweets."""
        e = cls.EMOJIS
        tweets = []

        # First tweet is the hook
        if content_type == "early_mail":
            venue_text = f"{venue} " if venue else ""
            first_tweet = f"{e['racing']} PUNTY'S {venue_text}TIPS\n\n"
        elif content_type == "results":
            first_tweet = f"{e['trophy']} RESULTS\n\n"
        else:
            first_tweet = f"{e['horse']} "

        # Split content into paragraphs/sections
        sections = content.split("\n\n")

        current_tweet = first_tweet
        for section in sections:
            # Check if section fits in current tweet
            if len(current_tweet) + len(section) + 2 <= cls.MAX_TWEET_LENGTH - 10:
                current_tweet += section + "\n\n"
            else:
                # Save current tweet and start new one
                if current_tweet.strip():
                    tweets.append(current_tweet.strip())
                current_tweet = section + "\n\n"

                # If single section is too long, split it
                if len(current_tweet) > cls.MAX_TWEET_LENGTH:
                    sentences = section.split(". ")
                    current_tweet = ""
                    for sentence in sentences:
                        if len(current_tweet) + len(sentence) + 2 <= cls.MAX_TWEET_LENGTH - 10:
                            current_tweet += sentence + ". "
                        else:
                            if current_tweet.strip():
                                tweets.append(current_tweet.strip())
                            current_tweet = sentence + ". "

        # Don't forget last tweet
        if current_tweet.strip():
            tweets.append(current_tweet.strip())

        # Add hashtags to last tweet
        if tweets:
            last = tweets[-1]
            if len(last) + len(hashtags) + 2 <= cls.MAX_TWEET_LENGTH:
                tweets[-1] = f"{last}\n\n{hashtags}"
            else:
                # Add hashtags as separate final tweet
                tweets.append(hashtags)

        # Limit thread length
        if len(tweets) > cls.THREAD_MAX_TWEETS:
            tweets = tweets[:cls.THREAD_MAX_TWEETS - 1]
            tweets.append(f"Full tips at punty.ai\n\n{hashtags}")

        return tweets


def format_twitter(
    raw_content: str,
    content_type: str = "early_mail",
    venue: Optional[str] = None,
) -> str:
    """Convenience function for formatting Twitter content."""
    return TwitterFormatter.format(raw_content, content_type, venue)


def format_twitter_thread(
    raw_content: str,
    content_type: str = "early_mail",
    venue: Optional[str] = None,
) -> list[str]:
    """Convenience function for formatting Twitter thread."""
    return TwitterFormatter.format_as_thread(raw_content, content_type, venue)
