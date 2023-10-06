import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
# Load model and image transforms
# parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
import string

charset_test = string.digits + string.ascii_lowercase
# if args.cased:
charset_test += string.ascii_uppercase
charset_test = "\u0020!\"#$%&\u0027[\\]^_`()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz{|}~£¥¦§­°±²³´·¹º×÷˙˚˝˶ΓΔΟΦμοπφ―‘’“”•‧′※₁₂₃₄₩€℃ℓ⅓ⅠⅡⅢⅣⅤⅥⅩ←↑→↓↔⇒⇔∅√∞∨∮∵≒≠≪≫⊙①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳ⒶⒷⒺⓐⓑⓒⓓⓔⓕⓖ─│┌┐└┘├■□▣▨▯▲△▶▷▼◀◁◆◇◈◉○◎●◐◑◦◯◼★☆☎☏☐☑☜☞♀♂♠♡♣♤♧✓✔⸢⸥。〃〈〉《》「」『』【】〓〔〕ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅋㅌㅍㅎㅒㅔㅗㅛㅡㅣㆍㆎ㈜㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛㉜㉝㉞㉟㉠㉡㉣㉮㉯㉰㉲㉵㉿㊱㊲㊳㊴㊵㎍㎎㎖㎗㎘㎜㎝㎞㎟㎠㎡㎢㎥㎦㏈丁他他代修切古地堂外對市年文日書替月月有洞海無田畓禮舊萬詩軍部金非面香가각간갇갈감갑값갓갔강갖갗같갚개객갤갯갱갸갹걍걔거걱건걷걸검겁것겄겅겉겊겋게겐겔겟겠겡겨격겪견겯결겸겹겻겼경곁계곘고곡곤곧골곰곱곳공곶과곽관괄괌광괘괜괴괸괼굉교굘굥구국군굳굴굵굼굽굿궁궂권궐궤귀귄규균귤그극근글금급긋긍긑긔긘기긱긴긷길김깁깃깅깆깉깊까깍깎깐깔깝깥깨깬깽꺼껀껄껏껑께껴꼈꼐꼬꼭꼰꼴꼼꽁꽂꽃꽉꾀꾸꾼꾿꿀꿈꿉꿔꿩뀌뀔뀜끄끈끊끌끓끔끕끗끙끝끼낀낄낌나낙낚난낟날낡남납났낭낮낯낱낳내낵낸낼냄냅냈냉냐냥너넉넌널넓넘넙넛넝넣네넥넨넬넷녀녁년념녕노녹논놀놈놋농높놓놘뇌뇐뇔뇨뇽누눅눈눌눔눕눙뉴뉼늄느늑는늘늠늣능늦늪늬니닉닌닐님닙닛닝다닥닦단닫달닭닳담답닷당닿대댁댄댈댐댑댓댕댖댜더덕던덜덟덤덧덩덮데덱덴델뎃뎅뎌뎐뎡도독돈돋돌돔돕돗동돠돨돼됐되된될됨됩됴두둑둔둘둠둡둥둬뒤뒷듀듈듕드득든듣들듬듭듯등듸듼딉디딕딘딜딤딥딧딩딪따딱딴딸땀땅때땐땔땜땡떠떡떤떨떳떻떼또똑똘똥뚜뚝뚫뛰뜨뜩뜬뜰뜸뜻띄띠띤라락란랄람랍랏랐랑랖래랙랜랠램랩랫랭랴략럇량러럭런럴럼럽렀렁렇레렉렌렐렘렛렝려력련렬렴렵렷렸령렺례로록론롤롬롭롯롱롸뢰료룐룔룡루룩룬룰룸룹룻룽뤄류륙륜률륨륭르륵른를름릅릇릉릐리릭린릴림립릿링마막만많맏말맑맘맙맛망맞맟맡매맥맨맵맹맺머먹먼멀멈멉멋멍멎메멕멘멜멤멧며멱면멸몃명몇몌모목몫몬몰몸몹못몽뫼묘무묵묶문묻물뭄뭉뭍뭐뮤뮨뮬뮴므믁믄믈믐미믹민믿밀밈밋밍밎및밑바박밖반받발밝밞밟밤밥방밭배백밴밸뱅뱌뱍버벅번벌범법벗벙벚베벡벤벨벼벽변별볌볍볏병볕보복볶본볼봄봅봇봉봐뵈부북분붇불붉붐붑붓붕붙뷔뷰브븍븐블븜븡븥비빅빈빋빌빔빕빗빙빚빛빠빨빵빼빽뺀뺌뺏뺑뺨뻘뼈뽀뽐뽑뽕뿌뿐뿔쁘쁜쁨쁩삐삘사삭산삳살삶삼삽삿상샅새색샌샐샘샛생샤샨샬샴샷샹서석섞선섣설섦섬섭섯성섶세섹센셀셈셉셋셔션셜셨셰소속손솔솜솝솟송솥솬쇄쇠쇳쇼숀숍숏숑수숙순술숨숩숫숭숯숲쉐쉘쉬쉰쉼쉽슁슈슐슘스슥슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싼쌀쌈쌍쌓쌰써썪썬썽쎄쎈쏘쏟쏠쏨쏩쑤쑥쑴쓰쓴쓸씀씁씅씌씨씩씬씰씸씻씽아악안앉않알앓암압앗았앙앞애액앤앨앰앱앳앵야약얀얄얇얏양얘어억언얻얼엄업없엇었엉엌엎에엑엔엘엠엣엥여역연열엷염엽엿였영옆예옌옐옛오옥온올옮옳옴옵옷옹옻와왁완왈왓왔왕왜외왼욉요욕욘욜욤욥용우욱운울움웁웃웅워웍원월웠웨웬웰웹위윈윌윗윙유육윤율윰융윷으윽은을음읍응의읜이익인읺일읽잃임입잆잇있잉잊잍잎자작잔잖잘잠잡잣장잦재잭잴쟁쟈쟝쟤저적전절젊점접젓정젖제젝젠젤젬젯져젼졀졌졍졔조족존졸좀좁좃종좋좌좡죄죌죠죤주죽준줄줌줍중줕줘줬쥐쥬쥰쥴즁즈즉즌즐즘즙증즤지직진질짐집짓징짖짙짚짜짝짧짱째쩨쩰쪼쪽쫄쫓쭈쭉쯔쯤찌찍찔찜차착찬찮찰참찹찻창찾채책챔챙챠처척천철첨첩첫청첮체첵첼쳋쳐쳔쳤쳬초촉촌촐촘촛총촤촬최추축춖춘출춤춥충춰췌취췽츄츈츠측츤츨츰층치칙친칠침칩칫칭카칵칸칼캄캅캇캐캑캔캘캠캡캣커컨컬컴컵컷컹케켄켐켓켔켙켜켠켰켱코콕콘콜콤콩콰쾌쾨쿄쿠쿡쿨쿵쿼퀘퀴퀵퀸퀼큐크큭큰클큼키킨킬킴킵킷킹타탁탄탈탐탑탓탕태택탠탤탬탭탯탱터턱턴털텀텁텃텅텉테텍텐텔템텝텡톄토톡톤톨톰톱통퇘퇴툐투툭툴툼퉁튀튄튜튱트특튼튿틀틈틉틍틔티틱틴틸팀팁팅파팍판팓팔팜팝팡패팩팬팰팽퍼펀펄펌펑펗페펙펜펠펫펴편펼폄폅평폐포폭폰폴폼폽퐁표푸푼풀품풍퓨퓰퓽프픅픈플픔픙피픽핀필핌핍핏핑하학한핟할함합핫항해핵핸햄햇했행햏햐햡향햬허헉헌헐험헙헛헝헤헨헬헹혀혁현혈혐협혓혔형혜호혹혼홀홈홉홍화확환활황회획횐횔횟횡효횬횽후훈훌훔훗훙훤훨훼휀휄휘휠휨휴휸휼흄흉흐흑흔흘흙흠흡흣흥흩희흰흴흽히힉힌힐힘힙힜힝念︎＂％＊＋＠［＼］｜｝～･￣￥￦他月無有案"

checkpoint="outputs/parseq/2022-10-24_00-36-26/checkpoints/last.ckpt"
device="cpu"

# if args.punctuation:
#     charset_test += string.punctuation
# kwargs.update({'charset_test': charset_test})
# print(f'Additional keyword arguments: {kwargs}')

parseq = load_from_checkpoint(checkpoint).eval().to(device)

# parseq =

img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

img = Image.open('demo_images/id3.jpg').convert('RGB')
# Preprocess. Model expects a batch of images with shape: (B, C, H, W)
img = img_transform(img).unsqueeze(0)

# transforms.extend([
#     T.Resize(img_size, T.InterpolationMode.BICUBIC),
#     T.ToTensor(),
#     T.Normalize(0.5, 0.5)
# ])
img=img.to(device)


logits = parseq(img)
logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

# Greedy decoding
pred = logits.softmax(-1)
label, confidence = parseq.tokenizer.decode(pred)
print('Decoded label = {}'.format(label[0]))