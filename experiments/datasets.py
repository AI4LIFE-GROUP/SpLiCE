from torchvision.datasets import CIFAR100, CIFAR10, Food101
from imagenetv2_pytorch import ImageNetValDataset
import os
import torch
import torchvision
from PIL import Image
import pandas as pd

class MITStates(torch.utils.data.Dataset):
    def __init__(self, path, train=True, transform=torchvision.transforms.ToTensor()):
        self.filepath = os.path.join(path, "mit_states/release_dataset/images/")

        self.transform = transform

        self.labels = {}
        self.class_to_idx = {}
        self.adj_to_idx = {}
        self.classadj_to_idx = {}
        counter = 0
        adj_counter = 0
        classadj_counter = 0
        test_idx = []

        with open(os.path.join(path,"mit_states/test_idx.txt"), "r") as f:
            lines = f.readlines()
            for l in lines:
                test_idx.append(int(l.strip()))

        split_idx = 0
        with open(os.path.join(path, "mit_states/mit_states_labels.csv"), "r") as f:
            lines = f.readlines()
            self.len = len(lines)
            for idx, line in enumerate(lines):

               
                if train:
                    if idx in test_idx:
                        continue
                else:
                    if idx not in test_idx:
                        continue

                if '"' in line.strip():
                    self.labels[split_idx] = line.strip().split('"')[0].split(",")[:-1] + [line.strip().split('"')[1]]
                else:
                    self.labels[split_idx] = line.strip().split(",")

                split_idx += 1

                if line.split(",")[0] not in self.class_to_idx:
                    self.class_to_idx[line.split(",")[0]] = counter
                    counter += 1
                
                if line.split(",")[1] not in self.adj_to_idx:
                    self.adj_to_idx[line.split(",")[1]] = adj_counter
                    adj_counter += 1

                if line.split(",")[0]+line.split(",")[1] not in self.classadj_to_idx:
                    self.classadj_to_idx[line.split(",")[0]+line.split(",")[1]] = classadj_counter
                    classadj_counter += 1



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = self.filepath + "/" + self.labels[idx][2]
        image = Image.open(path)
        label = self.class_to_idx[self.labels[idx][0]]
        adj = self.adj_to_idx[self.labels[idx][1]]
        classadj = self.classadj_to_idx[self.labels[idx][0]+self.labels[idx][1]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

class CelebA(torch.utils.data.Dataset):
    def __init__(self, path, train=True, transform=torchvision.transforms.ToTensor()):
        self.filepath = os.path.join(path, "celeba/")

        self.transform = transform
        self.img_paths = []
        self.labels = []


        # test_idx = []

        male_no_glasses, female_no_glasses, male_glasses, female_glasses = 0, 0, 0, 0
        self.class_to_idx = {
            "female": 0,
            "male": 1,
        }
        # self.class_to_idx = {0: [], 1: [], 2: [], 3: []}
        with open(os.path.join(path,"celeba/list_attr_celeba.csv"), "r") as f:
            lines = f.readlines()
            self.len = len(lines)
            for idx, line in enumerate(lines):
                line = line.strip()
                line = line.split(",")

                img_path = line[0]
                black_hair, blonde_hair, brown_hair, gray_hair = line[9], line[10], line[12], line[18]
                male, young, glasses, hat = line[21], line[40], line[16], line[36]


                # if young == "-1" and gray_hair == "1":

                if male == "1" and glasses == "-1":
                    if male_no_glasses <= 74:
                        if train:
                            self.img_paths.append(img_path)
                            self.labels.append(0)
                            # self.class_to_idx[0].append(img_path)
                    else:
                        if not train and (male_no_glasses <= 149):
                            self.img_paths.append(img_path)
                            self.labels.append(0)
                            # self.class_to_idx[0].append(img_path)
                    male_no_glasses += 1

                elif male == "-1" and glasses == "-1":
                        if female_no_glasses <= 74:
                            if train:
                                self.img_paths.append(img_path)
                                self.labels.append(1)
                                # self.class_to_idx[1].append(img_path)
                        else:
                            if not train and (female_no_glasses <= 149):
                                self.img_paths.append(img_path)
                                self.labels.append(1)
                                # self.class_to_idx[1].append(img_path)
                        female_no_glasses += 1

                elif male == "1" and glasses == "1":
                    if male_glasses <= 74:
                        if train:
                            self.img_paths.append(img_path)
                            self.labels.append(2)
                            # self.class_to_idx[2].append(img_path)
                    else:
                        if not train and (male_glasses <= 149):
                            self.img_paths.append(img_path)
                            self.labels.append(2)
                            # self.class_to_idx[2].append(img_path)
                    male_glasses += 1
                
                elif male == "-1" and glasses == "1":
                    if female_glasses <= 74:
                        if train:
                            self.img_paths.append(img_path)
                            self.labels.append(3)
                            # self.class_to_idx[3].append(img_path)
                    else:
                        if not train and (female_glasses <= 149):
                            self.img_paths.append(img_path)
                            self.labels.append(3)
                            # self.class_to_idx[3].append(img_path)
                    female_glasses += 1        
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = os.path.join(self.filepath, "img_align_celeba/", self.img_paths[idx])
        image = Image.open(path)

        label = self.labels[idx]
        genderlabel = 1 if ((label == 1) or (label == 3)) else 0
        glasseslabel = 1 if (label >= 2) else 0
        if self.transform is not None:
            image = self.transform(image)
       
        return image, genderlabel#, glasseslabel
    
class WaterbirdDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform, train=True):

        self.dataset_name = "waterbird_complete95_forest2water2"
        self.dataset_dir = os.path.join(data_path, self.dataset_name)
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.') 
        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'metadata.csv'))
        if train:
            self.metadata_df = self.metadata_df[self.metadata_df['split']==0]
        else:
            self.metadata_df = self.metadata_df[self.metadata_df['split']==2]

        self.y = self.metadata_df['y'].values
        self.places = self.metadata_df['place'].values
        self.filenames = self.metadata_df['img_filename'].values
        self.transform = transform

        self.class_to_idx = {
            "landbirdonland": 0,
            "waterbirdonland": 1,
            "waterbirdbirdonland": 3,
            "landbirdonwater": 2,
        }

        print(len(self.y))
        print(len(self.filenames))

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        y = self.y[idx]
        place = self.places[idx]
        img_filename = os.path.join(
            self.dataset_dir,
            self.filenames[idx])
        img = Image.open(img_filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, y+2*place#, place

def load(dataset, preprocess, data_path, train=False):
    download = False

    if data_path == None:
        os.makedirs("./datasets/", exist_ok=True)
        download = True
        data_path = "./datasets/"

    if dataset == "CIFAR10":
        dataset_test = CIFAR10(data_path, download=download, train=train, transform=preprocess)

    elif dataset == "CIFAR100":
        dataset_test = CIFAR100(data_path, download=download, train=train, transform=preprocess)

    elif dataset == "MITStates":
        dataset_test = MITStates(data_path,train=train, transform=preprocess)

    elif dataset == "CelebA":
        dataset_test = CelebA(data_path, train=train, transform=preprocess)
    
    elif dataset == "Waterbirds":
        dataset_test = WaterbirdDataset(data_path, train=train, transform=preprocess)

    elif dataset == "Food101":
        dataset_test = Food101(data_path, split="test", download=download, transform=preprocess)

    elif dataset == "ImageNetVal":
        dataset_test = ImageNetValDataset(transform=preprocess, location=data_path)

        dataset_test.class_to_idx = {}

        classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "projectile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglass", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

        for i in range(1000):
            dataset_test.class_to_idx[classes[i]] = i

    else:
        raise RuntimeError(f"Dataset {dataset} not supported.")

    return  dataset_test