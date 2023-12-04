apt update
apt install -y vim git
git clone https://github.com/1p22geo/tensorflow-chat.git
cd tensorflow-chat
ssh-keygen
git config --global user.email "1p22geodecki@gmail.com"
git config --global user.name "1p22geo"
echo Copy the following key to github:
echo "========================"
cat ~/.ssh/id_rsa.pub
echo "========================"

read -n 1
git remote set-url origin git@github.com:1p22geo/tensorflow-chat.git
git pull

for (( ;; )) do python lstm_seq2seq.py ; git reset --soft HEAD~; git add .; git commit -m"."; git push origin master --force; done;
