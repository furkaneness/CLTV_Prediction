##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################


##############################################################
# 1. Data Preperation
##############################################################
import pandas as pd
import numpy as np
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
import matplotlib.pyplot as plt
from lifetimes.plotting import plot_period_transactions

pd.set_option("display.max_columns", None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def outlier_thresholds(dataframe, variable):
    q1 = dataframe[variable].quantile(0.01)
    q3 = dataframe[variable].quantile(0.99)
    interquantile_range = q3-q1
    up_limit = q3 + 1.5 * interquantile_range
    low_limit = q1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df_ = pd.read_excel("C:/Users/EnesKurban/Desktop/VBO Bootcamp 8.Dönem/Datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.shape
df.info()
df.describe().T

df = df[~df["Invoice"].str.contains("C", na=False)] # Fatura Numarasında "C" harfi bulunanlar iade olanlar.
df.isnull().sum()
df.dropna(inplace=True)
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Price"] * df["Quantity"]
df = df[df["Country"] == "United Kingdom"]
df.info()


df["InvoiceDate"].max() # Son alışveriş tarihi 2011-12-09 olarak bulundu. 2 gün sonrasını analiz tarihi olsun.
today_date = dt.datetime(2011, 12, 11)

#############################################
# RFM Table
#############################################

# Recency, frequency ve T değerlerini parametre olarak alır.
# Recency   = Müşterinin son alışveriş tarihi - ilk alışveriş tarihi(Haftalık)
# Frequency = Müşteri toplamda kaç alışveriş yaptı.
# Monetary  = Müşterinin alışveriş başına ortalama harcadığı para
# T         = Müşterinin yaşı (analiz tarihi - ilk alışveriş tarihi)(Haftalık)

rfm = df.groupby("Customer ID").agg({"InvoiceDate": [lambda date: (date.max()-date.min()).days,
                                                     lambda date: (today_date-date.min()).days],
                                     "Invoice"    :  lambda num: num.nunique(),
                                     "TotalPrice" :  lambda TotalPrice: TotalPrice.sum()})

rfm.columns = rfm.columns.droplevel(0)
rfm.columns = ["recency", "T", "frequency", "monetary"]

# BG-NBD için AVG_MONETARY, WEEKLY RECENCY VE WEEKLY T'nin HESAPLANMASI
rfm["monetary"] = rfm["monetary"] / rfm["frequency"]
rfm["T"] = rfm["T"] / 7
rfm["recency"] = rfm["recency"] / 7

# freq > 1
rfm = rfm[rfm["frequency"] > 1]

##############################################################
# 2. BG/NBD Modelinin Kurulması
##############################################################

# pip install lifetimes

# BG-NBD Modeli : Expected Number of Transaction (Beklenen İşlem Sayısı)
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(rfm["frequency"],
        rfm["recency"],
        rfm["T"])

################################################################
# 6 Ay İçerisinde Beklenen Satın Alma Sayıları
################################################################

bgf.predict(24,
            rfm["frequency"],
            rfm["recency"],
            rfm["T"])

rfm["expected_purc_6_month"] = bgf.predict(24,
                                           rfm["frequency"],
                                           rfm["recency"],
                                           rfm["T"])

################################################################
# 6 Ayda Tüm Şirketin Beklenen Satış Sayısı Nedir?
################################################################

bgf.predict(24,
            rfm["frequency"],
            rfm["recency"],
            rfm["T"]).sum()

################################################################
# Tahmin Sonuçlarının Değerlendirilmesi
################################################################

plot_period_transactions(bgf)
plt.show()

##############################################################
# 3. GAMMA-GAMMA Modelinin Kurulması
##############################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(rfm["frequency"], rfm["monetary"])

rfm["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm["frequency"], rfm["monetary"])

##############################################################
# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
##############################################################

cltv = ggf.customer_lifetime_value(bgf,
                                   rfm["frequency"],
                                   rfm["recency"],
                                   rfm["T"],
                                   rfm["monetary"],
                                   freq="W",
                                   time=6,
                                   discount_rate=0.01)
cltv.head()
cltv = cltv.reset_index()
cltv_final = rfm.merge(cltv, on="Customer ID", how="left")
cltv_final.head()
cltv_final.describe().T


################################################################
# 5. Farklı Zaman Periyotlarından Oluşan CLTV Analizi
################################################################

################################################################
# 1 Aylık CLTV Hesaplanması
################################################################

cltv_1 = ggf.customer_lifetime_value(bgf, rfm["frequency"], rfm["recency"], rfm["T"], rfm["monetary"], freq="W", time=1, discount_rate=0.01)
cltv_1 = cltv_1.reset_index()
cltv_final = cltv_final.merge(cltv_1, on="Customer ID", how = "left")
cltv_final.rename(columns={'clv_x': 'clv_6_month'}, inplace=True)
cltv_final.rename(columns={'clv': 'clv_1_month'}, inplace=True)

################################################################
# 12 Aylık CLTV Hesaplanması
################################################################

cltv_2 = ggf.customer_lifetime_value(bgf, rfm["frequency"], rfm["recency"], rfm["T"], rfm["monetary"], freq="W", time=12, discount_rate=0.01)
cltv_2 = cltv_2.reset_index()
cltv_2.rename(columns={'clv': 'clv_12_month'}, inplace=True)
cltv_final = cltv_final.merge(cltv_2, on="Customer ID", how = "left")


################################################################
# 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişinin analizi
################################################################

cltv_final.sort_values(by="clv_1_month", ascending=False).head(10)
cltv_final.sort_values(by="clv_12_month", ascending=False).head(10)


################################################################
# 6. Segmentasyon
################################################################

cltv_final["SEGMENT"] = pd.qcut(cltv_final["clv_6_month"], 4, labels=["D", "C", "B", "A"])
