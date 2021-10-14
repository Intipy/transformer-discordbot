import re
import matplotlib.pyplot as plt
import tensorflow as tf
from Transformer.train import START_TOKEN, END_TOKEN, MAX_LENGTH, model, tokenizer

import discord
from discord.ext import commands
from discord.utils import get

import random
import numpy as np
import time



model.load_weights('C:/Users/gio/Downloads/trans/Save/weight_')




def preprocess_sentence(sentence):
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = sentence.strip()
  return sentence



def evaluate(sentence):
  sentence = preprocess_sentence(sentence)

  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  # 디코더의 예측 시작
  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)

    # 현재(마지막) 시점의 예측 단어를 받아온다.
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    # 마지막 시점의 예측 단어를 출력에 연결한다.
    # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)



def predict(sentence):
  prediction = evaluate(sentence)

  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  print('Input: {}'.format(sentence))
  print('Output: {}'.format(predicted_sentence))

  return predicted_sentence
########################
client = commands.Bot(command_prefix=")")



@client.event
async def on_ready():
    await client.change_presence(activity=discord.Game(name=")도움말"))
    print("Bot Ready")



@client.event
async def on_message(message):
    if message.author.bot: 
        return None
    
    
    if message.content.startswith(")"):
        
        
        if message.content == ")도움말":
            embed=discord.Embed(title="឵ ឵ ឵឵ ឵ ឵឵ ឵ ឵឵ ឵ ឵", color=0x00ff56)
            embed.set_author(name="[ 파이드 도움말 ]", icon_url="https://cdn.discordapp.com/attachments/851075781413437474/851339042448605224/1.png")
            embed.add_field(name="឵파이드는 AI 챗봇입니다", value="- ឵ ឵឵ ឵ `prefix: )`", inline=False) 
            embed.add_field(name="឵) 뒤에 하고싶은 말을 해보세요", value="- ឵ ឵឵ ឵ ឵`)안녕`", inline=False) 
            embed.add_field(name="파이드는 현재 영어를 지원하지 않습니다", value="- ឵ ឵឵ ឵ ឵`한글만 사용해주세요`", inline=False) 
            embed.add_field(name="឵파이드는 계속 학습 중입니다", value="`개발 초기 단계이며 아직 학습량이 적어 여러 오류나 부적절한 행동이 있을 수 있습니다(학습 상태를 참조해주세요)`", inline=False) 
            embed.add_field(name="개발자", value="- ឵ ឵឵ ឵ `Intipy#1111`", inline=False) 
            embed.add_field(name="개발 일자", value="- ឵ ឵឵ ឵ `2021.06.05.`", inline=False)  
            embed.add_field(name="파이드 공식 지원 서버", value="- ឵ ឵឵ ឵ https://discord.gg/yUQDS2sWZH", inline=False) 
            embed.add_field(name="학습 상태", value="- ឵ ឵឵ ឵ `0.00000032 (매우 약함)`", inline=False)
            embed.add_field(name="메세지 관리", value="- ឵ ឵឵ ឵ `)도움말 메세지`", inline=False) 
            await message.channel.send(embed=embed)
            
            
            
        elif message.content == ")도움말 메세지":
            embed=discord.Embed(title="឵ ឵ ឵឵ ឵ ឵឵ ឵ ឵឵ ឵ ឵", color=0x00ff56)
            embed.set_author(name="[ 파이드 메세지 관리 도움말 ]", icon_url="https://cdn.discordapp.com/attachments/851075781413437474/851339042448605224/1.png")
            embed.add_field(name="현재 채널에서 메세지 개수만큼 삭제", value="- ឵ ឵឵ ឵ `)cn <개수>`", inline=False)
            embed.add_field(name="현재 채널에서 메세지 내용으로 삭제하기", value="- ឵ ឵឵ ឵ `)ck <내용>`", inline=False)
            embed.add_field(name="현재 채널에서 특정 유저 메세지 삭제하기", value="- ឵ ឵឵ ឵ `)cn <유저 멘션>`", inline=False)
            #embed.add_field(name="현재 채널에서 특정 유저 메세지 실시간으로 삭제(해당 명령어를 사용한 시점부터 중지 명령어를 사용하기 전까지 해당 유저가 쓰는 글이 실시간으로 삭제됩니다)", value="- ឵ ឵឵ ឵ `)ct <유저 멘션>`", inline=False) 
            #embed.add_field(name="실시간 삭제 중지 ", value="- ឵ ឵឵ ឵ `)cs <개수>`", inline=False)
            await message.channel.send(embed=embed)
            
            
            
            
        elif message.content.startswith(")ck"):
            #675635343952838658
            if message.author.guild_permissions.manage_messages:
                try:
                    splited_keyword = message.content.split(" ")[1]
                except:
                    await message.channel.send("올바른 문자를 입력해주세요")
                await message.delete()
                
                if message.guild:
                    async for message in message.channel.history():
                        if message.content in splited_keyword:
                            try:
                                await message.delete()
                            except:
                                pass
            else:
                await message.channel.send("해당 명령어를 사용할 권한이 없습니다")
                  
                  
                  
        elif message.content.startswith(")cn"):
            if message.author.guild_permissions.manage_messages:
                try:
                    splited_number = int(message.content.split(" ")[1])
                except:
                    await message.channel.send("올바른 개수를 입력해주세요")
                await message.delete()
                await message.channel.purge(limit=splited_number)
            else:
                await message.channel.send("해당 명령어를 사용할 권한이 없습니다")
                
                
                
        elif message.content.startswith(")cu"):
            if message.author.guild_permissions.manage_messages:
                try:
                    splited_user = message.mentions[0].id
                except:
                    await message.channel.send("올바른 유저를 입력해주세요")
                await message.delete()
                await message.channel.purge(limit=1000, check=lambda m: m.author.id == splited_user)
            else:
                await message.channel.send("해당 명령어를 사용할 권한이 없습니다")
            

        
        else:
            uMsg = message.content[1:]
            res = predict(uMsg)
            await message.channel.send(res)

            


client.run("NzkwODQ2MTM1MzIxMjMxMzgx.X-Gi2w.MxfU9m8caRA6rV4EHgrfyIe8Kzo")

