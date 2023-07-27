from django.urls import path

from . import views

urlpatterns = [
    # 定义 path 和 方法  的  关系
    path("", views.index, name="index"),
    path("questionList", views.question_list, name="question_list"),
    path("questionDetail/<int:question_id>", views.question_detail, name="question_detail")
]