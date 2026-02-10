"""Tests for Facebook delivery integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from punty.formatters.facebook import FacebookFormatter, format_facebook


# ── Formatter Tests ──────────────────────────────────────────────────────────


class TestFacebookFormatter:
    """Test FacebookFormatter markdown cleaning and output."""

    def test_strips_markdown_headers(self):
        text = "### MEET SNAPSHOT\nSome content"
        result = FacebookFormatter.format(text)
        assert "###" not in result
        assert "MEET SNAPSHOT" in result

    def test_strips_numbered_section_headers(self):
        text = "### 2) MEET SNAPSHOT\nContent here"
        result = FacebookFormatter.format(text)
        assert "###" not in result
        assert "2)" not in result
        assert "MEET SNAPSHOT" in result

    def test_strips_bold_markers(self):
        text = "**Bold text** and *italic text*"
        result = FacebookFormatter.format(text)
        assert "**" not in result
        assert "*" not in result
        assert "Bold text" in result
        assert "italic text" in result

    def test_strips_underscore_markers(self):
        text = "__bold__ and _italic_"
        result = FacebookFormatter.format(text)
        assert "__" not in result
        assert "bold" in result
        assert "italic" in result

    def test_collapses_extra_newlines(self):
        text = "Line 1\n\n\n\n\nLine 2"
        result = FacebookFormatter.format(text)
        assert "\n\n\n" not in result
        assert "Line 1\n\nLine 2" in result

    def test_adds_gambling_footer(self):
        result = FacebookFormatter.format("Some tips content")
        assert "Gamble Responsibly" in result
        assert "1800 858 858" in result

    def test_no_duplicate_gambling_footer(self):
        text = "Content\n\nGamble Responsibly. gamblinghelponline.org.au | 1800 858 858"
        result = FacebookFormatter.format(text)
        assert result.count("Gamble Responsibly") == 1

    def test_truncates_over_limit(self):
        text = "x" * 70000
        result = FacebookFormatter.format(text)
        assert len(result) <= FacebookFormatter.MAX_LENGTH
        assert "Full tips at punty.ai" in result

    def test_headers_uppercased(self):
        text = "## Race by Race Tips"
        result = FacebookFormatter.format(text)
        assert "RACE BY RACE TIPS" in result

    def test_convenience_function(self):
        result = format_facebook("### Hello World")
        assert "###" not in result
        assert "HELLO WORLD" in result


# ── Delivery Tests ───────────────────────────────────────────────────────────


class TestFacebookDeliveryConfigured:
    """Test FacebookDelivery.is_configured()."""

    @pytest.mark.asyncio
    async def test_not_configured_without_keys(self):
        from punty.delivery.facebook import FacebookDelivery

        db = AsyncMock()
        fb = FacebookDelivery(db)

        with patch("punty.delivery.facebook.FacebookDelivery._load_keys") as mock_load:
            async def set_no_keys():
                fb._page_id = None
                fb._access_token = None
                fb._keys_loaded = True
            mock_load.side_effect = set_no_keys

            assert not await fb.is_configured()

    @pytest.mark.asyncio
    async def test_configured_with_both_keys(self):
        from punty.delivery.facebook import FacebookDelivery

        db = AsyncMock()
        fb = FacebookDelivery(db)

        with patch("punty.delivery.facebook.FacebookDelivery._load_keys") as mock_load:
            async def set_keys():
                fb._page_id = "123456"
                fb._access_token = "EAAtoken123"
                fb._keys_loaded = True
            mock_load.side_effect = set_keys

            assert await fb.is_configured()

    @pytest.mark.asyncio
    async def test_not_configured_missing_token(self):
        from punty.delivery.facebook import FacebookDelivery

        db = AsyncMock()
        fb = FacebookDelivery(db)

        with patch("punty.delivery.facebook.FacebookDelivery._load_keys") as mock_load:
            async def set_partial():
                fb._page_id = "123456"
                fb._access_token = None
                fb._keys_loaded = True
            mock_load.side_effect = set_partial

            assert not await fb.is_configured()


class TestFacebookDeliverySend:
    """Test FacebookDelivery.send() with mocked httpx."""

    @pytest.mark.asyncio
    async def test_send_posts_to_graph_api(self):
        from punty.delivery.facebook import FacebookDelivery

        # Mock DB with content + meeting
        db = AsyncMock()
        mock_content = MagicMock()
        mock_content.id = "test-content-1"
        mock_content.status = "approved"
        mock_content.raw_content = "### Tips\nGreat tips here"
        mock_content.facebook_formatted = None
        mock_content.content_type = "early_mail"
        mock_content.meeting_id = "test-meeting-1"

        mock_meeting = MagicMock()
        mock_meeting.venue = "Flemington"

        # DB execute returns content then meeting
        call_count = 0
        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalar_one_or_none.return_value = mock_content
            else:
                result.scalar_one_or_none.return_value = mock_meeting
            return result
        db.execute = mock_execute
        db.commit = AsyncMock()

        fb = FacebookDelivery(db)
        fb._page_id = "123456"
        fb._access_token = "EAAtoken123"
        fb._keys_loaded = True

        # Mock httpx response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "123456_789"}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await fb.send("test-content-1")

        assert result["status"] == "posted"
        assert result["post_id"] == "123456_789"
        mock_client.post.assert_called_once()
        # Verify the URL contains the page ID
        call_args = mock_client.post.call_args
        assert "123456/feed" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_send_raises_if_not_configured(self):
        from punty.delivery.facebook import FacebookDelivery

        db = AsyncMock()
        fb = FacebookDelivery(db)
        fb._page_id = None
        fb._access_token = None
        fb._keys_loaded = True

        with pytest.raises(ValueError, match="Facebook not configured"):
            await fb.send("test-content-1")

    @pytest.mark.asyncio
    async def test_send_raises_if_content_not_found(self):
        from punty.delivery.facebook import FacebookDelivery

        db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        db.execute = AsyncMock(return_value=mock_result)

        fb = FacebookDelivery(db)
        fb._page_id = "123456"
        fb._access_token = "EAAtoken123"
        fb._keys_loaded = True

        with pytest.raises(ValueError, match="Content not found"):
            await fb.send("nonexistent")

    @pytest.mark.asyncio
    async def test_send_raises_if_not_approved(self):
        from punty.delivery.facebook import FacebookDelivery

        db = AsyncMock()
        mock_content = MagicMock()
        mock_content.status = "draft"
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_content
        db.execute = AsyncMock(return_value=mock_result)

        fb = FacebookDelivery(db)
        fb._page_id = "123456"
        fb._access_token = "EAAtoken123"
        fb._keys_loaded = True

        with pytest.raises(ValueError, match="Content not approved"):
            await fb.send("test-content-1")


class TestFacebookDeliveryDelete:
    """Test FacebookDelivery.delete_post()."""

    @pytest.mark.asyncio
    async def test_delete_calls_graph_api(self):
        from punty.delivery.facebook import FacebookDelivery

        db = AsyncMock()
        fb = FacebookDelivery(db)
        fb._page_id = "123456"
        fb._access_token = "EAAtoken123"
        fb._keys_loaded = True

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.delete.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await fb.delete_post("123456_789")

        assert result["status"] == "deleted"
        assert result["post_id"] == "123456_789"


class TestFacebookDeliveryComment:
    """Test FacebookDelivery.post_comment()."""

    @pytest.mark.asyncio
    async def test_post_comment_on_post(self):
        from punty.delivery.facebook import FacebookDelivery

        db = AsyncMock()
        fb = FacebookDelivery(db)
        fb._page_id = "123456"
        fb._access_token = "EAAtoken123"
        fb._keys_loaded = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "123456_789_comment1"}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await fb.post_comment("123456_789", "Great win!")

        assert result["status"] == "commented"
        assert result["comment_id"] == "123456_789_comment1"
        assert result["parent_post_id"] == "123456_789"
        # Verify URL includes /comments
        call_args = mock_client.post.call_args
        assert "123456_789/comments" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_post_comment_raises_if_not_configured(self):
        from punty.delivery.facebook import FacebookDelivery

        db = AsyncMock()
        fb = FacebookDelivery(db)
        fb._page_id = None
        fb._access_token = None
        fb._keys_loaded = True

        with pytest.raises(ValueError, match="Facebook not configured"):
            await fb.post_comment("123456_789", "Hello")

    @pytest.mark.asyncio
    async def test_post_comment_raises_on_api_error(self):
        from punty.delivery.facebook import FacebookDelivery

        db = AsyncMock()
        fb = FacebookDelivery(db)
        fb._page_id = "123456"
        fb._access_token = "EAAtoken123"
        fb._keys_loaded = True

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"error": {"message": "Invalid post ID"}}
        mock_response.text = "Invalid post ID"

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ValueError, match="Facebook comment error"):
                await fb.post_comment("bad_id", "Hello")


class TestAutoPostToFacebook:
    """Test automation.auto_post_to_facebook()."""

    @pytest.mark.asyncio
    async def test_skips_if_not_configured(self):
        from punty.scheduler.automation import auto_post_to_facebook

        db = AsyncMock()

        with patch("punty.delivery.facebook.FacebookDelivery.is_configured", return_value=False):
            result = await auto_post_to_facebook("test-content", db)

        assert result["status"] == "skipped"
        assert "not configured" in result["reason"]

    @pytest.mark.asyncio
    async def test_returns_error_on_exception(self):
        from punty.scheduler.automation import auto_post_to_facebook

        db = AsyncMock()

        with patch("punty.delivery.facebook.FacebookDelivery.is_configured", return_value=True), \
             patch("punty.delivery.facebook.FacebookDelivery.send", side_effect=Exception("API down")):
            result = await auto_post_to_facebook("test-content", db)

        assert result["status"] == "error"
        assert "API down" in result["message"]
