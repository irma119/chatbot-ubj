# Chatbot Deployment Unichat

referensi [this](https://github.com/python-engineer/pytorch-chatbot) tutorial menggunakan Flask dan JavaScript.

## Download Aplikasi 
![visual studio code](https://code.visualstudio.com/download)
![miniconda](https://docs.anaconda.com/free/miniconda/index.html)
![anaconda](https://docs.anaconda.com/free/anaconda/install/windows/)
![ngrok](https://ngrok.com/)

# Daftar akun
![Twilio](https://login.twilio.com/u/signup?state=hKFo2SBHRTNSc091cVZTUjN1ZUdLOEZ2ODAydkJ1WW1ocEtPUKFur3VuaXZlcnNhbC1sb2dpbqN0aWTZIGNDbmVCTUJXTDBtZmE0Qm5fajB4VzBsckpJTHgtVUY0o2NpZNkgTW05M1lTTDVSclpmNzdobUlKZFI3QktZYjZPOXV1cks)

Jika aplikasi sudah di download dan sudah buat akun Twilio silahkan anda dapat menginstall semua aplikasi. 
1. Pada aplikasi visual studio code anda mengharuskan mendowload extensions python, html, css, dan jupiter
2. Untuk aplikasi miniconda atau anaconda silahkan download library ( torch, json, numpy, nltk, matplotlib seaborn dan flask)
Selanjutnya Install package flask
```
$ (base)pip install Flask torch torchvision nltk
```
Setelah selesai, Install nltk package
```
$ (base) python
>>> import nltk
>>> nltk.download('punkt')
>>> ctrl+z/quit
```
4. Pada aplikasi ngrok silahkan daftar akun anda, aplikasi ini merupakan server untuk menjalankan aplikasi chatbot whatsapp
5. Silahkan daftar akun pada aplikasi twilio untuk mndapatkan API Whatsapp

## Running aplikasi 
Pada kode terdapat 2 file python yaitu app.py dan app2.py
1. app.py
Pada aplikasi yang pertama merupakan chatbot PMB UBJ dengan implemntasi pada website untuk menjalankan aplikasi tersebut dilakukan dengan cara sebagai berikut:
Running file app.py, kemudian pada terminal akan muncul [http://5000 ](http://127.0.0.1:5000) silahkan klik untuk menampilkan aplikasi, jika berhasil akan tampil seperti berikut
![image](https://github.com/irma119/chatbot-ubj/assets/110200862/62f03bf7-50f2-4c97-9ff2-6b0b2aebd5be)
Jalankan aplikasi dengan klik icon chat di pojok kanan bawah. Ketikan pertanyaan sesuai dengan data pada json
Untuk menghentikan program silahkan klik ctrl+c

3. app2.py
Untuk running pada file app2.py ini anda harus mendapatkan SID, token dan no whatsapp dari twilio untuk mendapatkannya anda dapat memilih messaging pada website twilio di bagian sender whatsapp
![image](https://github.com/irma119/chatbot-ubj/assets/110200862/5ae00f62-11fb-42f8-90f1-56e42761faf3)
silahkan ikuti langkah yang diperintahkan pada langkah kedua anda akan mndapatkan SID, token dan nowhatsapp untuk chatbot.
Jika sudah selesai silahkan copy paste nomor tersebut pada file **env.konf** untuk dapat menjalankan aplikasi chatbot pada whatsapp

Selanjutnya anda buka aplikasi ngrok dengan copy kode token "ngrok config add-authtoken 2bSbKvuoeAZpqQvL3kQBd1m6sdA_3gMy263Cf8sSRH1jAz6QM" ( kode ini akan brbeeda pada setiap akun jadi silahkan cek pada akun yag terdaftar.
Silahkan paste kode tersebut pada aplikasi ngrok dan tekan enter 
jika sudah berhasil silahkan ketikkan "ngrok http 5000" untuk mengaktifkan server, server yang berhasil terhubung akan tampil sebagai berikut :
![image](https://github.com/irma119/chatbot-ubj/assets/110200862/b6004448-da56-4583-964b-8662ea816f47)

jika sudah selesai silahkan running file **app2.py** untuk menghubungkan server kedalam whatsapp
selanjutnya anda dapat copy http **web interface ** untuk di paste pada akun twilio anda 

![image](https://github.com/irma119/chatbot-ubj/assets/110200862/64ff108b-a196-4a8a-a230-66f6ce996627)
silahkan anda dapat paste http tadi pada bagian **when a message comes in** dengan method **POST**

Untuk mengaktifkan aplikasi silahka buka whatsapp dengan mengirim pesan **private-anyone** ke nomor whatsapp twilio. Jika berhasil anda akan mendapatkan pesan "Twilio Sandbox: ✅ You are all set! The sandbox can now send/receive messages from whatsapp:+14155238886. Reply stop to leave the sandbox any time."

Silahkan masukkan pesan sesuai dengan data pada json.
contoh chat pada whatsapp
![image](https://github.com/irma119/chatbot-ubj/assets/110200862/b07788d1-7aa4-4117-b61c-ffac8bc58896)

*Selamat mencoba semoga bermanfaat 
Note: "DIHARAPKAN SEMUA FILE DIBUAT SATU FOLDER JIKA INGIN MENDOWNLOAD FOLDER DIATAS SILAHKAN DI BAGIAN POJOK KANAN PILIH DOWNLOAD ZIP"

![ini](https://github.com/irma119/chatbot-deployment-unichat/assets/110200862/0e5745ef-f27a-4e93-a60d-de3b30438b7f)




