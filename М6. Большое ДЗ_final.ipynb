# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 23:29:45 2022

@author: Игорь
"""


import pandas as pd
train = pd.read_csv('data/train.csv',
                   usecols=[0,1, 2, 3, 4, 5, 7, 8, 9], # импортируем данные и присваиваем им требуемые типы
                   dtype={
                          'row_id': 'category',
                          'timestamp': 'int32',
                          'user_id': 'category',
                          'content_id': 'category',
                          'content_type_id': 'category',
                          'task_container_id': 'category',
                          'user_answer': 'int8',
                          'answered_correctly':'int8',
                          'prior_question_elapsed_time': 'category',
                          'prior_question_had_explanation': 'boolean'
                          },
                   )
train_df = train.copy(deep = True)

lectures = pd.read_csv('data/HW/lectures.csv')
lectures_df = lectures.copy(deep = True)                

questions = pd.read_csv('data/HW/questions.csv')
questions_df = questions.copy(deep = True)

print(train_df)

#%%
pd.set_option('display.max_columns', 20)

print(train_df.head(5))
print(train_df.info())
print(train_df.describe())

"""Список параметров для анализа:
    1. Общее кол-во студентов
    2. Самые частозадаваемые вопросы
    3. Вопрос с самым большимс кол-вом правильных ответв
    4. Вопрос с самым малым кол-вом правильных ответов
    5. Студент с самой высокой успеваемостью
    6. Студент с самой низкой успеваемостью
    7. Средний % верных ответов на вопросы
    8. Время, затраченное на лекции.   

    """
    
#%% 1. Общее кол-во студентов
    
stud_qty = train_df['user_id'].nunique()
st_list = train_df['user_id'].value_counts() # Перечень студентов. Потребуется позже
print('Общее кол-во студентов:', stud_qty)


#%%  2.    Самые частозадаваемые вопросы

quest_df = train_df[train_df['content_type_id'] == '0'] #создаём таблицу с информацией только по вопросам (не по лекциям). вопрос, если 'content_type_id' == 0
quest_list = quest_df['content_id'].value_counts()    # 
print('Самые частозадаваемые вопросы:', quest_list.head(100), '\n')

#%%      3. Вопрос с самым большим кол-вом правильных ответов

correct_answ = pd.DataFrame({ #создаём таблицу с 4 колонками (id вопроса, кол-во правильных вопросов, кол-во попыток ответа, доля правильных ответов)
    'question_id:': [],
    'correct_answ:': [],
    'all_answ:': [],
    '% of correct answ:': []})
q_lst = set(quest_list.index) # создаём список индексов всех вопросов
for i in q_lst:
    a = quest_df[quest_df['content_id'] == i].shape[0] # сколько раз был задан  данный вопарос
    b = quest_df[(quest_df['content_id'] == i) & (quest_df['answered_correctly'] == 1)].shape[0]  # в скольких случаях на вопрос был дан правильный ответ
    if a < 50: # учитывапется только статистика по вопросам, заданным более 50 раз
        continue
    else:
        new_row = {'question_id:': i,
                   'correct_answ:': b,
                   'all_answ:': a,
                   '% of correct answ:': b/a
                   }
        correct_answ = correct_answ.append(new_row, ignore_index=True)
print(correct_answ)

# %% 3. Вопрос с самым большимс кол-вом правильных ответов. Продолжение...
print(correct_answ.sort_values('% of correct answ:', ascending = False)) # сверху - самые лёгкие вопросы



# %%  4. Вопрос с самым малым кол-вом правильных ответов
print(correct_answ.sort_values('% of correct answ:')) # сверху - самые сложные вопросы




# %%     5. Студент с самой высокой успеваемостью

stud_rate = pd.DataFrame({
    'user_id:': [],
    'correct_answ:': [],
    'all_answ:': [],
    '% of correct answ:': []})
stud_lst = set(st_list.index) # создаём список id всех студентов
for i in stud_lst:
    a = quest_df[quest_df['user_id'] == i].shape[0] # сколько вопросов было задано студенту
    b = quest_df[(quest_df['user_id'] == i) & (quest_df['answered_correctly'] == 1)].shape[0]  # в скольких случаях на вопрос был дан правильный ответ
    if a == 0: 
        continue
    else:
        new_row = {'user_id:': i,
                   'correct_answ:': b,
                   'all_answ:': a,
                   '% of correct answ:': b/a
                   }
        stud_rate = stud_rate.append(new_row, ignore_index=True)
print(stud_rate)

# %%     5. Студент с самой высокой успеваемостью. Продолжение...
print(stud_rate.sort_values('% of correct answ:', ascending = False)) # сверху - студенты с самой высокой успеваемостью

# %%     6. Студент с самой низкой успеваемостью. Продолжение...
print(stud_rate.sort_values('% of correct answ:', ascending = True)) # сверху - студенты с самой низкой успеваемостью

#%% 7. Средний % верных ответов на вопросы
a = stud_rate['correct_answ:'].sum() # получено правильныйх ответов
b = stud_rate['all_answ:'].sum()  # всего вопросов было задано
print('% правильных ответов для всех студентов:', round(a/b*100,1))


#%% 8. Время, затраченное на лекции.

#Создаём таблицу с новым столюцом 'spent_time', в который заносим продолжительность каждогго этапа обучения (лекции ил ответа на вопрос)

for i in range(len(train_df)-1):
    if train_df.iloc[int(i+1)]['timestamp'] != 0:
        a = train_df.iloc[int(i)]['timestamp']
        b = train_df.iloc[int(i)+1]['timestamp']
        train_df.at[i,'spent_time'] =  b - a #заносим в ячейку разность между временем окончания и временем начала лекции
    else:
        continue
    
    #print(train_df['timestamp'].loc[train_df.index[int(i)+1]])
#%% 8. Время, затраченное на лекции. Продолжение...


lect_df = train_df[train_df['content_type_id'] == '1'] #создаём таблицу с информацией только по лекциям (не по вопросам). 
print(lect_df)

lect_list = lect_df['content_id'].value_counts()    # Количество проведённых лекций с данным id
print('Самые частые лекции:', lect_list.head(100), '\n') #печатаем id первых 100 наиболее популярных лекций


print('Средрее время, затраченное на лекции, минут:', round(lect_df['spent_time'].mean() / 60000))  # считаем и печатаем среднее время, затраченное на 1 лекцию, в минутах

