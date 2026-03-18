from clip_video.config import BrandConfig


class TestSocialPlatforms:
    def test_default_social_platforms(self):
        config = BrandConfig(name="test")
        assert config.social_platforms == ["linkedin"]

    def test_custom_social_platforms(self):
        config = BrandConfig(name="test", social_platforms=["linkedin", "youtube"])
        assert "youtube" in config.social_platforms

    def test_serialization_roundtrip(self):
        config = BrandConfig(name="test", social_platforms=["linkedin", "youtube"])
        d = config.model_dump()
        restored = BrandConfig(**d)
        assert restored.social_platforms == ["linkedin", "youtube"]

    def test_missing_field_backwards_compatible(self):
        config = BrandConfig.model_validate({"name": "test"})
        assert config.social_platforms == ["linkedin"]
