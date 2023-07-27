python -m pip search django
python -m pip install django
python -m django --version
# 在workspace目录 创建一个django项目
django-admin startproject my_django_site
# 进入django项目
cd .\my_django_site\
# 启动项目
python .\manage.py runserver
# 在项目中新增一个app
python .\manage.py startapp polls
# 数据模型变更后，对具体app生成模型迁移版本信息
python .\manage.py makemigrations polls
# 显示迁移版本的sql  app 版本号
python .\manage.py sqlmigrate polls 0001
# 对所有app生成模型迁移版本信息
python .\manage.py makemigrations
python .\manage.py sqlmigrate polls 0002
# 将模型变更应用到数据库
python .\manage.py migrate
python .\manage.py runserver
# 创建管理员
python .\manage.py createsuperuser
python .\manage.py runserver
