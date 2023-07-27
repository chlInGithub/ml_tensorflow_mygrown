from django.shortcuts import render
from django.http import HttpResponse
from .models import Question
from django.http import Http404,JsonResponse
from django.core import serializers
import logging
from django.views.decorators.cache import cache_page
from django.core import cache
import json


logging.getLogger().setLevel(level='DEBUG')


def index(request):
    return HttpResponse("hello world")

# Create your views here.


def question_list(request):
    query_set = Question.objects.all()[:5]
    result = APIResult(data=query_set.values('id'))
    # {"code": 200, "data": [{"id": 1, "question_text": "xxxxxxxxxxx?", "pub_date": "2023-07-25T07:05:08Z"}, {"id": 2, "question_text": "\u554a\u554a\u554a\u554a\u554a", "pub_date": "2023-07-27T02:00:59Z"}]}
    return JsonResponse(result, safe=False)

# 缓存方法
#@cache_page(20)
def question_detail(request, question_id):
    # 使用cache函数
    c = cache.cache.get(f'question:{question_id}')
    logging.info(f"cache data is {c}")
    if c is None:
        cache.cache.set(f"question:{question_id}", 'cache data')

    logging.info(f"question detail {question_id}")
    logging.error(f"question detail {question_id}")
    try:
        #query_set = Question.objects.filter(pk=question_id)
        query_set = Question.objects.all()
    except Question.DoesNotExist:
        raise Http404("No question")

    result = APIResult(data=query_set.values()[0])
    #print(json.dumps(result))
    # {"code": 200, "data": {"id": 1, "question_text": "xxxxxxxxxxx?", "pub_date": "2023-07-25T07:05:08Z"}}
    return JsonResponse(result)

    #return HttpResponse(serializers.serialize("json", query_set))


class APIResult(dict):
    def __init__(self, data):
        self['code'] = 200
        if isinstance(data, dict):
            print("data is dict")
            self['data'] = data
        elif isinstance(data, list):
            print("data is list")
            self['data'] = data
        else:
            print("data is not dict or list")
            self['data'] = list(data)

