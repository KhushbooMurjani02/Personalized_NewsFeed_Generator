import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from tqdm import tqdm, tqdm_notebook
from functools import reduce
import os


def getSources():
    source_url = 'https://newsapi.org/v1/sources?language=en'
    response = requests.get(source_url).json()
    sources = dict()

    for source in response['sources']:
        dct=source['id']
        sources[dct]=source['category']
        
    return sources

def mapping():
    d = {}
    response = requests.get('https://newsapi.org/v1/sources?language=en')
    response = response.json()
    i=0
    for s in response['sources']:
        lst=[]
        lst.append(s['category'])
        lst.append(s['name'])
        lst.append(s['description'])
        d[i]=lst
        i+=1
    return d

def category(source, m):
    try:
        return m[source]
    except:
        return 'NC'

def getDailyNews():
    yt=0
    surc = getSources()
    sources=[]
    sources=surc.keys()
    
    key='55feabfcd61c4f3e916b9f7cf24ba048'
    url = 'https://newsapi.org/v1/articles?source={0}&sortBy={1}&apiKey={2}'

    responses = []
    final=[]
    bus=0
    tec=0
    spt=0
    ent=0
    sci=0
    grl=0
    ttl_lst=[]
    for i, source in tqdm_notebook(enumerate(sources), total=len(sources)): 
        
        # if surc[source] =='business':
        #     key='6dbb708e222a459ebec065c3e2066f77'
        # elif surc[source]=='technology':
        #     key='54e59984b5b24a2d842697cc7c90bbaa'
        # elif surc[source]=='science':
        #     key='b4a1bbaa682341eb8b4abc517f2f8b18'
        # elif surc[source]=='sports':
        #     key='909518b5f3d44f9682eb3a3a3da65b70'
        # elif surc[source]=='entertainment':
        #     key='828126b9d5ce49c397f56a616c0d56a3'
        # else:
        #     key='59f46ecf5a5b4eefae15db1398f4a587'
        
        u = url.format(source, 'top', key)
        v= url.format(source, 'latest', key)
               
        response = requests.get(u)
        r = response.json()

        response = requests.get(v)
        s=response.json()
        # print(r,"\n\n")
        flag=''
        fname=''
        files=open('123.txt','w')
        try:         
            for article in r['articles']:

                # print("ARTICLES IS:",article,"\n\n")
                ttl=article['title']
                descp=article['description']
                # print("title: ",ttl,"\n\ndescp: ",descp)
                # print(bus)
                # bus+=1
                if (ttl is None) or (descp is None):
                        pass
                else:
                    if ttl not in ttl_lst:
                        # print("title: ",ttl,"\n\ndescp: ",descp)
                        f=surc[source]
                        #print(f)
                        if f=="business":
                            fname="g:/final_project/bbc/"+f+"/"+str(bus)+".txt"
                            bus+=1
                            flag='b'
                        elif f=='technology':
                            fname="g:/final_project/bbc/"+f+"/"+str(tec)+".txt"
                            tec+=1
                            flag='t'
                        elif f=='sports':
                            fname="g:/final_project/bbc/"+f+"/"+str(spt)+".txt"
                            spt+=1
                            flag='sp'
                        elif f=='entertainment':
                            fname="g:/final_project/bbc/"+f+"/"+str(ent)+".txt"
                            ent+=1
                            flag='e'
                        elif f=='science':
                            fname="g:/final_project/bbc/"+f+"/"+str(sci)+".txt"
                            sci+=1
                            flag='sc'
                        else:
                            fname="g:/final_project/bbc/"+f+"/"+str(grl)+".txt"
                            grl+=1
                            flag='g'
                            
                        files = open(fname,'w')
                        # print(fname)
                        # print("before")
                        art=ttl+'\n'+descp
                        ttl_lst.append(ttl)
                        files.write(art)
                        files.close()

        except Exception as e:
            files.close()
            # print(fname)
            os.remove(fname)
            if flag=='b':
                bus-=1
            elif flag=='t':
                tec-=1
            elif flag=='sp':
                spt-=1
            elif flag=='e':
                ent-=1
            elif flag=='g':
                grl-=1
            else:
                sci-=1
            
            
            
            
            # print(e)
            
     
    # articles = list(map(lambda r: r['articles'], responses))
    # articles = list(reduce(lambda x,y: x+y, articles))
    
    # news = pd.DataFrame(articles)
    # news = news.dropna()
    # news = news.drop_duplicates()
    # news.reset_index(inplace=True, drop=True)
    # jink = mapping()
    
   
    # print(news)
    # news['category'] = news['source'].map(lambda s: category(s, d))
    # news['scraping_date'] = datetime.now()
    # i=2
    # print("news")

    # jink=dict()
    # for u,v in d.items():
    #     jink[u]=v
    # print(jink)
    # try:
    #     aux = pd.read_excel('news.xlsx')
    #     aux = aux.append(news)
    #     print("hello..")
    #     print(aux)
    #     aux = aux.drop_duplicates('url')
    #     aux.reset_index(inplace=True, drop=True)
    #     aux.to_excel('news.xlsx', encoding='utf-8', index=False)
    # except:
    #     news.to_excel('news.xlsx', index=False, encoding='utf-8')
        
    print('Done')
    
if __name__=='__main__':
    getDailyNews()
