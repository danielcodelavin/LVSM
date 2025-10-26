import sys,json,requests
import getopt

if sys.version_info[0] < 3:
    raise Exception("Python 3 or greater is required. Try running `python3 download_collection.py`")

collection_name = ''
owner_name = ''

# Read options
optlist, args = getopt.getopt(sys.argv[1:], 'o:c:')

sensor_config_file = ''
private_token = ''
for o, v in optlist:
    if o == "-o":
        owner_name = v.replace(" ", "%20")
    if o == "-c":
        collection_name = v.replace(" ", "%20")

if not owner_name:
    print('Error: missing `-o <owner_name>` option')
    quit()

if not collection_name:
    print('Error: missing `-c <collection_name>` option')
    quit()


print("Downloading models from the {}/{} collection.".format(owner_name, collection_name.replace("%20", " ")))

page = 1
count = 0


base_url ='https://fuel.gazebosim.org/'


fuel_version = '1.0'
next_url = '/models?page={}&per_page=100&q=collections:{}'.format(page,collection_name)


download_url = base_url + fuel_version + '/{}/models/'.format(owner_name)


while True:
    url = base_url + fuel_version + next_url

   
    r = requests.get(url)

    if not r or not r.text:
        break

   
    models = json.loads(r.text)

  
    page = page + 1
    next_url = '/models?page={}&per_page=100&q=collections:{}'.format(page,collection_name)
  
    
    for model in models:
        count+=1
        model_name = model['name']
        print ('Downloading (%d) %s' % (count, model_name))
        download = requests.get(download_url+model_name+'.zip', stream=True)
        with open(model_name+'.zip', 'wb') as fd:
            for chunk in download.iter_content(chunk_size=1024*1024):
                fd.write(chunk)

print('Done.')
