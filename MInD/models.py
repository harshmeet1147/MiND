from django.utils.timezone import now    
from django.db import models

#all var is in snakecase

class Sku30Days(models.Model):
    date = models.DateField()
    keyword = models.CharField(max_length=255)
    asin = models.CharField(max_length=255)
    title = models.CharField(max_length=510)
    brand = models.CharField(max_length=255)
    mrp = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    page_result = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    url = models.URLField(max_length=1275)
    category_dashb = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.keyword} - {self.asin}"
    


class SearchFrequencyRank(models.Model):
    search_freq_rank = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    search_term = models.CharField(max_length=255, null=True, blank=True)

    top_clicked_brand_1 = models.CharField(max_length=255, null=True, blank=True)
    top_clicked_brand_2 = models.CharField(max_length=255, null=True, blank=True)
    top_clicked_brand_3 = models.CharField(max_length=255, null=True, blank=True)

    top_clicked_category_1 = models.CharField(max_length=255, null=True, blank=True)
    top_clicked_category_2 = models.CharField(max_length=255, null=True, blank=True)
    top_clicked_category_3 = models.CharField(max_length=255, null=True, blank=True)

    top_clicked_asin_1 = models.CharField(max_length=255, null=True, blank=True)
    top_clicked_title_1 = models.TextField(null=True, blank=True)
    top_clicked_product_clickshare_1 = models.FloatField(null=True, blank=True)
    top_clicked_product_convshare_1 = models.FloatField(null=True, blank=True)

    top_clicked_asin_2 = models.CharField(max_length=255, null=True, blank=True)
    top_clicked_title_2 = models.TextField(null=True, blank=True)
    top_clicked_product_clickshare_2 = models.FloatField(null=True, blank=True)
    top_clicked_product_convshare_2 = models.FloatField(null=True, blank=True)

    top_clicked_asin_3 = models.CharField(max_length=255, null=True, blank=True)
    top_clicked_title_3 = models.TextField(null=True, blank=True)
    top_clicked_product_clickshare_3 = models.FloatField(null=True, blank=True)
    top_clicked_product_convshare_3 = models.FloatField(null=True, blank=True)

    date = models.DateField(null=True, blank=True)
    category_dashb = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return f"{self.search_term} - {self.search_freq_rank}"



class Coverage(models.Model):
    date = models.DateField(null=True, blank=True)
    keyword = models.CharField(max_length=255, null=True, blank=True)
    title = models.TextField(null=True, blank=True)
    is_sponsored = models.CharField(max_length=255, null=True, blank=True)
    slot = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    page_result = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    asin = models.CharField(max_length=255, null=True, blank=True)
    brand = models.CharField(max_length=255, null=True, blank=True)
    category_dashb = models.CharField(max_length=255, null=True, blank=True)
    hour = models.CharField(max_length=255, null=True, blank=True)
    clicks = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    sku_visibility = models.FloatField(null=True, blank=True)
    sku_visibility_index = models.FloatField(null=True, blank=True)
    sku_index = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.date} - {self.keyword} - {self.asin}"
    

class SentimentalAnalysis(models.Model):
    asin = models.CharField(max_length=255, null=True, blank=True)
    date = models.DateField(null=True, blank=True)
    reviews = models.TextField(null=True, blank=True)
    ratings = models.FloatField(null=True, blank=True)
    compound = models.FloatField(null=True, blank=True)
    analysis = models.CharField(max_length=255, null=True, blank=True)
    brand = models.CharField(max_length=255, null=True, blank=True)
    category_dashb = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return f"{self.asin} - {self.date}"



class GV(models.Model):
    brand_name = models.CharField(max_length=255, null=True, blank=True)
    date = models.DateField(null=True, blank=True)
    n2bGVlookBackPeriod = models.DecimalField(max_digits=10, decimal_places=0, null=True, blank=True)
    parentAsin = models.CharField(max_length=255, null=True, blank=True)
    parentAsinTitle = models.CharField(max_length=255, null=True, blank=True)
    childAsin = models.CharField(max_length=255, null=True, blank=True)
    childAsinTitle = models.CharField(max_length=255, null=True, blank=True)
    category_dashb = models.CharField(max_length=255, null=True, blank=True)
    subcategory = models.CharField(max_length=255, null=True, blank=True)
    indexedGlanceViews = models.DecimalField(max_digits=20, decimal_places=0, null=True, blank=True)
    indexedNewToBrandGVs = models.DecimalField(max_digits=20, decimal_places=0, null=True, blank=True)
    newToBrandGVsPercentage = models.FloatField(null=True, blank=True)
    adsShareOfNewToBrandGVsPercent = models.FloatField(null=True, blank=True)
    asinConversion = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return f"{self.brand_name} - {self.date}"




class HeliumData(models.Model):
    asin = models.CharField(max_length=255, null=True, blank=True)
    title = models.CharField(max_length=1020, null=True, blank=True)
    brand = models.CharField(max_length=255, null=True, blank=True)
    price = models.FloatField(null=True, blank=True)
    sales = models.FloatField(null=True, blank=True)
    revenue = models.FloatField(null=True, blank=True)
    bsr = models.DecimalField(max_digits=20, decimal_places=0, null=True, blank=True)
    active_sellers = models.CharField(max_length=255, null=True, blank=True)
    ratings = models.FloatField(null=True, blank=True)
    reviews_count = models.DecimalField(max_digits=20, decimal_places=0, null=True, blank=True)
    date = models.DateField(null=True, blank=True)
    category = models.CharField(max_length=255, null=True, blank=True)
    category_dashb = models.CharField(max_length=255, null=True, blank=True)
    rank = models.BigIntegerField(null=True, blank=True)

    def __str__(self):
        return f"{self.title[:50]} - {self.asin}"


class HeliumDataTitleTokens(models.Model):
    asin = models.TextField(null=True, blank=True)
    title = models.TextField(null=True, blank=True)
    brand = models.TextField(null=True, blank=True)
    price = models.FloatField(null=True, blank=True)
    sales = models.FloatField(null=True, blank=True)
    revenue = models.FloatField(null=True, blank=True)
    bsr = models.FloatField(null=True, blank=True)
    active_sellers = models.TextField(null=True, blank=True)
    ratings = models.FloatField(null=True, blank=True)
    reviews_count = models.FloatField(null=True, blank=True)
    date = models.DateField(null=True, blank=True)
    category = models.TextField(null=True, blank=True)
    category_dashb = models.TextField(null=True, blank=True)
    rank = models.BigIntegerField(null=True, blank=True)
    cleaned_title = models.TextField(null=True, blank=True)
    tokenized_title = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.asin} - {self.title[:50]}"


class sov(models.Model):
    keyword = models.CharField(max_length=1020)
    keyword_type = models.CharField(max_length=255, blank=True, null=True)
    category = models.CharField(max_length=255, blank=True, null=True)
    sub_category = models.CharField(max_length=255, blank=True, null=True)
    min_traffic = models.DecimalField(max_digits=20, decimal_places=2, blank=True, null=True)
    max_traffic = models.DecimalField(max_digits=20, decimal_places=2, blank=True, null=True)
    category_dashb = models.CharField(max_length=255, blank=True, null=True)
    date = models.DateField(blank=True, null=True)

    def __str__(self):
        return self.keyword


class FlipkartToken(models.Model):
    user_id = models.CharField(max_length=100)
    access_token = models.TextField()
    refresh_token = models.TextField()
    expires_in = models.IntegerField()
    refresh_expires_in = models.IntegerField()
    token_type = models.CharField(max_length=20)
    scope = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)

    