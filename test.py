import pandas as pd

df_evidences = pd.DataFrame({
    "claim": ["馬丁·路德·金恩在1998年4月4日，於美國田納西州孟菲斯旅館內遭人暗殺去世。",
              "馬丁·路德·金恩在1998年4月4日，於美國田納西州孟菲斯旅館內遭人暗殺去世。",
              "馬丁·路德·金恩在1998年4月4日，於美國田納西州孟菲斯旅館內遭人暗殺去世。",
              "馬丁·路德·金恩在1998年4月4日，於美國田納西州孟菲斯旅館內遭人暗殺去世。"],
    "text": ["小馬丁 · 路德 · 金恩 （ Martin Luther King , Jr. ) 是一位美國牧師 、 社會運動者 、 人權主義者和非裔美國人民權運動領袖 ， 也是1964年諾貝爾和平獎得主 。",
             "他主張以非暴力的公民抗命方法爭取非裔美國人的基本權利 ， 而成爲的象徵 。",
             "馬丁 · 路德 · 金出生時名爲麥可 · 金 （ Michael King ） ， 他的父親爲了紀念16世紀歐洲宗教改革領袖馬丁 · 路德而將他改名爲小馬丁 · 路德 · 金 。",
             "身爲一位浸信會牧師 ， 金在他職業生涯早期就已開始投入民權運動 ， 曾領導1955年聯合抵制蒙哥馬利公車運動 ， 並在1957年協助建立 （ SCLC ） 。"],
    "evidence" : [[[6203,6030,"馬丁·路德·金",18],[6203,6030,"馬丁·路德·金恩遇刺案",0]],
                  [[6203,6030,"馬丁·路德·金",18],[6203,6030,"馬丁·路德·金恩遇刺案",0]],
                  [[6203,6030,"馬丁·路德·金",18],[6203,6030,"馬丁·路德·金恩遇刺案",0]],
                  [[6203,6030,"馬丁·路德·金",18],[6203,6030,"馬丁·路德·金恩遇刺案",0]]],
    "predicted_evidence": [["馬丁·路德·金",0], ["馬丁·路德·金",1], ["馬丁·路德·金",4], ["馬丁·路德·金",5]],
    "prob": [0.2, 0.3, 0.5, 0.8]
})
top_n = 4

print(df_evidences)

top_rows = (
    df_evidences.where(df_evidences["prob"].gt(0.9)).groupby("claim", group_keys=True).apply(
    lambda x: x.nlargest(top_n, "prob"))
    .reset_index(drop=True)
)
print(top_rows)

if top_rows.empty == True:
    top_rows = (
        df_evidences.groupby("claim").apply(
        lambda x: x.nlargest(top_n, "prob"))
        .reset_index(drop=True)
    )
    print(top_rows)

claim = "馬丁·路德·金恩在1998年4月4日，於美國田納西州孟菲斯旅館內遭人暗殺去世。"
predicted_evidence = top_rows[top_rows["claim"] == claim]["predicted_evidence"].tolist()
print(predicted_evidence)