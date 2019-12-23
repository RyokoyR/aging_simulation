#!/lustre7/home/lustre4/ryoyokosaka/python/.pyenv/shims
import sys
sys.path.append('/lustre7/home/lustre4/ryoyokosaka/python/')

import pandas as pd
import numpy as np
import category_encoders as ce

rnaseq_df = pd.read_table('/Users/yokosaka/Desktop/python/data/1_54rnaseq_drop.txt',index_col = 0)
#rnaseq_df = pd.read_table('/lustre7/home/lustre4/ryoyokosaka/python/',index_col=0)

#rnaseqデータフレームから年齢情報の抽出
age_Seriese = rnaseq_df.index.str[-2:]
age_label_df = pd.DataFrame(age_Seriese,index = rnaseq_df.index,columns = ["age_label"])
#年齢カテゴリの種類は
print(age_label_df["age_label"].nunique())

#年齢カテゴリをワンホットベクトル化
#エンコードしたい列を指定
list_cols = ["age_label"]
#ワンホットエンコードしたい列を指定して変換
ce_ohe = ce.OneHotEncoder(cols=list_cols,handle_unknown='ignore')
#データフレームを渡して変換する
age_label_oh_df = ce_ohe.fit_transform(age_label_df)
#onehotencoding完了
age_label_oh_df

#Ordinalencoding 順序ベクトル化
ce_oe = ce.OrdinalEncoder(cols=list_cols,handle_unknown='ignore')
age_label_oe_df = ce_oe.fit_transform(age_label_df)
age_label_oe_df["age_label"]
