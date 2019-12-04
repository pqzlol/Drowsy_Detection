#coding=utf-8
class Driver:
    #定义属性
    #总帧数
    totalFrame=0
    #时间周期闭眼帧数
    closeFrame=0
    #眼睛闭合占帧比
    ECR=0.0
    #眨眼频率（帧数）
    BF=0
    #最长持续闭眼时间
    maxECT=0.0
    #点头频率
    NF=0
    #打哈欠频率
    YF=0
    #打哈欠持续最长时间
    maxYT=0.0
    #疲劳值
    fatigueRation=0.0
    
    #构造函数
    def __init__(self):
        self.init()
    
    #初始化函数
    def init(self):
        #总帧数
        self.totalFrame=0
        #眼睛闭合数
        self.closeFrame=0
        #眼睛闭合占帧比
        self.ECR=0.0
        #眨眼频率（帧数）
        self.BF=0
        #最长持续闭眼时间
        self.maxECT=0.0
        #点头频率
        self.NF=0
        #打哈欠频率
        self.YF=0
        #打哈欠持续最长时间
        self.maxYT=0.0
        #疲劳值
        self.fatigueRation=0.0