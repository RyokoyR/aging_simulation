#!/lustre7/home/lustre4/ryoyokosaka/python/.pyenv/shims
import sys
sys.path.append('/lustre7/home/lustre4/ryoyokosaka/python/')

import pandas as pd
import glob
import re
import pandas as pd
#処理するファイル名のリストを取得
#スパコン側のアドレス
path = glob.glob("/lustre7/home/lustre4/ryoyokosaka/python/data/GSE41080/*_probe.txt")
#遺伝子IDが入っているファイルのみ別に読み取る
#path.remove("/Users/yokosaka/Desktop/python/data/GSE41080/GSM1008309_93173_7280_probe.txt")

#データテーブルをみると一つの転写産物にいくつかのプローブが該当しているように見える。
#今回はこれらを平均化することとする。

#かく転写産物が一回ずつ現れるように前処理する。
#SYMBOL行で重複のある転写産物名のリストをえる。
data_ID_exist = pd.read_table("/lustre7/home/lustre4/ryoyokosaka/python/data/GSE41080/GSM1008309_93173_7280_probe.txt",index_col="PROBE_ID")
SYMBOL = pd.DataFrame(data_ID_exist["SYMBOL"])
SYMBOL = SYMBOL.rename(columns = {'SYMBOL':'Symbol'})
SYMBOL_bool = SYMBOL.duplicated()
SYMBOL_duplicated = SYMBOL_bool[SYMBOL_bool == True]
#重複プローブを持つ転写物名リストを取得。
duplicated_transcript = list(SYMBOL.loc[SYMBOL_duplicated.index]['Symbol'].unique())

GSE41080_expression_df = pd.DataFrame()
for txt_file in path:
    data = pd.read_table(txt_file,index_col="PROBE_ID")
    signal_name = re.findall('\w\w\w\w\w\w\w\w\w\w_\w.AVG_Signal',str(data.columns))[0]
    data = data.rename(columns = {signal_name:'Signal'})
    data = pd.concat([data,SYMBOL],axis = 1)
    #ファイルパスからサンプルめい(GSM...を取得する)
    pattern = 'GSM\w\w\w\w\w\w\w'
    sample_name = re.findall(pattern,txt_file)


    #重複プローブの平均値を計算し新しいデータフレームに格納
    #新しいデータフレームを用意
    _expression_df = pd.DataFrame()
    list_read_SYMBOL = []

    for symbol in list(data['Symbol']):
        #もしプローブidが重複プローブであるならば平均値を計算する
        if symbol in duplicated_transcript:
            mean = data.loc[data.Symbol == symbol,'Signal'].mean()
            low = pd.DataFrame(mean,index=[symbol],columns=['Signal'])
            _expression_df = pd.concat([_expression_df,low])
        #もしプローブidが重複していないのであればシグナルはそのまま
        elif symbol not in list_read_SYMBOL:
            #今参照しているインデックス名をindに入れる
            ind = list(data[data["Symbol"] == symbol].index)
            low = pd.DataFrame(float(data['Signal'].loc[ind]),index=[symbol],columns=['Signal'])
            _expression_df = pd.concat([_expression_df,low])
        #すでに重複シンボルに対して平均値を計算していた場合そのループを飛ばす。
        elif symbol in list_read_SYMBOL:
            continue



        #読んだシンボルリストに追加
        list_read_SYMBOL.append(symbol)
    _expression_df = _expression_df.rename(columns = {"Signal":"sample_name"})
    GSE41080_expression_df = pd.concat([GSE41080_expression_df,_expression_df],axis=1)
    print(_expression_df)
print(GSE41080_expression_df)
GSE41080_expression_df.to_csv("/lustre7/home/lustre4/ryoyokosaka/python/data/GSE41080/GSE41080_expression.csv")

