from Bio import Entrez
import pandas as pd
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
import time
from selenium.webdriver.chrome.options import Options


def get_url(search_term):
    # Uncomment the below line if using locally without dockerizing
    # driver = webdriver.Chrome()
    # Uncomment the below five lines if deploying it using docker
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(chrome_options=chrome_options)
    driver.get('https://www.ncbi.nlm.nih.gov/mesh')
    # Fill the search term
    button_elem = driver.find_element_by_id('term')
    button_elem.send_keys(search_term)
    # Click on the search button
    button_elem = driver.find_element_by_id('search')
    button_elem.click()
    url_to_return = driver.current_url
    driver.close()
    return url_to_return


def get_entry_terms(url):
    page = requests.get(url)
    all_list = list()

    soup = BeautifulSoup(page.content, 'html.parser')
    try:
        try:
            # This is used to get the Entry terms if the link is direct
            results = soup.find('div', class_='rprt abstract').findChild('ul', recursive=False).find_all('li')
            for res in results:
                # Check if the content inside contains HTML tags or not
                if res.find() is None:
                    all_list.append(res.contents[0])
        except:
            # This is used to get the possible search terms if the link is not direct
            results = soup.findAll('div', class_='rprt')
            for res in results:
                content = res.find('p', class_='title').find('a').contents[0]
                href = "https://www.ncbi.nlm.nih.gov" + res.find('p', class_='title').find('a')['href']
                return href, content
    except:
        # If there are no Entry terms
        return all_list, None
    return all_list, None


class dataDownload():

    def __init__(self, entries_per_class):
        self.df = pd.DataFrame()
        Entrez.email = "siddharthss500@gmail.com"
        Entrez.api_key = "7d993523c56b78396869d4d063f1ede4ec08"
        self.entries_per_class = entries_per_class

    def _search(self, query):
        # Search for the UIDs based on the query
        handle_s = Entrez.esearch(
            db="pubmed", retmax=10, sort="relevance", term=(query), field="abstract")
        ids = Entrez.read(handle_s)["IdList"]
        time.sleep(1)
        # Fetch the documents based on the UIDs
        handle_f = Entrez.efetch(
            db="pubmed", id=ids, rettype="abstract", retmode="xml")
        results = Entrez.read(handle_f)
        return results

    def _fetch(self, results):
        temp_df = pd.DataFrame()
        rank, pmid, title, abstract = [[] for i in range(4)]
        for idx, artcle in enumerate(results['PubmedArticle']):
            try:
                abst = artcle['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                # Check if abstract is empty or not
                if not abst:
                    continue
                abstract.append(abst)
                id = artcle['MedlineCitation']['PMID']
                pmid.append(id)
                tit = artcle['MedlineCitation']['Article']['ArticleTitle']
                title.append(tit)
                rank.append(idx)
            except:
                continue

        # Fill dataframe
        temp_df['Rank'] = rank
        temp_df['PMID'] = pd.Series(pmid, dtype='str')
        temp_df['Title'] = title
        temp_df['Abstract'] = abstract
        return temp_df

    def _filter(self, search_terms):
        temp_df = pd.DataFrame()
        for term in search_terms:
            results = self._search(term)
            temp_df = pd.concat([temp_df, self._fetch(results)])
        # Groupby PMID and sum up the ranks
        temp_df = temp_df.groupby(
            ["PMID", "Title", "Abstract"]).sum().reset_index()
        temp_df["Rank"] = temp_df["Rank"] / len(search_terms)
        temp_df = temp_df.sort_values(
            by=["Rank"], ascending=False).reset_index()
        return temp_df

    def _checkisnotin(self, lst1, lst2):
        # len(lst1) + len(lst2) == len(set(lst1.PMID).symmetric_difference(lst2.PMID))
        if lst1.PMID.isin(lst2.PMID).sum() == 0:
            return 0
        return list(set(lst1.PMID).symmetric_difference(lst2.PMID))

    def _splitdataset(self, dframe, num):
        return dframe.iloc[:num, :], dframe.iloc[num + 1:, :]

    def _createlabel(self, dframe, lab1, lab2):
        dframe["term_one"] = lab1
        dframe["term_two"] = lab2
        return dframe

    def createdata(self, current_entry_terms):
        # For class 1
        class_one_df = self._filter(current_entry_terms[0])
        # For class 2
        class_two_df = self._filter(current_entry_terms[1])
        # Check if there is anything common between the two classes
        if self._checkisnotin(class_one_df, class_two_df) == 0:
            pass
        else:
            lst_to_rmve = self._checkisnotin(class_one_df, class_two_df)
            # For class one
            to_retain = list(set(class_one_df.PMID.tolist()).difference(set(lst_to_rmve)))
            class_one_df = class_one_df.loc[class_one_df['PMID'].isin(to_retain)]
            # For class two
            to_retain = list(set(class_two_df.PMID.tolist()).difference(set(lst_to_rmve)))
            class_two_df = class_two_df.loc[class_two_df['PMID'].isin(to_retain)]
            print("Note that a common of ", len(lst_to_rmve), " IDs were removed")
        # For both the classes
        # We will combine take the first eighteen terms of each class and use zip (AND)
        terms_to_take = min(len(current_entry_terms[0]), len(current_entry_terms[1]))
        search_terms_three = list(map(
            lambda i: i[0] + " AND " + i[1],
            zip(current_entry_terms[0][:terms_to_take], current_entry_terms[1][:terms_to_take])))
        class_both_df = self._filter(search_terms_three)

        # Create the final dataset
        # For class 1
        class_one_df, dframe1 = self._splitdataset(class_one_df, self.entries_per_class)
        class_one_df = self._createlabel(class_one_df, 1, 0)
        self.df = pd.concat([self.df, class_one_df])
        # For class 2
        class_two_df, dframe2 = self._splitdataset(class_two_df, self.entries_per_class)
        class_two_df = self._createlabel(class_two_df, 0, 1)
        self.df = pd.concat([self.df, class_two_df])
        # For both
        class_both_df, dframe3 = self._splitdataset(class_both_df, self.entries_per_class)
        class_both_df = self._createlabel(class_both_df, 1, 1)
        self.df = pd.concat([self.df, class_both_df])
        # Combine dframe1 and dframe1; randomize
        dframe1 = pd.concat([dframe1, dframe2, dframe3])
        dframe1 = dframe1.sample(frac=1).reset_index()
        dframe1, _ = self._splitdataset(dframe1, self.entries_per_class)
        dframe1 = self._createlabel(dframe1, 0, 0)
        self.df = pd.concat([self.df, dframe1])

        # Drop unnecessary columns
        self.df = self.df.drop(["index", "Rank", "level_0"], axis=1)
        return


def create_entry_terms(terms):
    two_terms = terms
    two_entry_sets = list()
    new_term = list()
    which_term = None
    count = 0
    for trm in two_terms:
        current_url = get_url(search_term=trm)
        current_entry_terms, updated_term = get_entry_terms(current_url)
        if updated_term is None:
            two_entry_sets.append(current_entry_terms)
        else:
            # Take the closest search term
            # Send back the new URL to get the corresponding entry terms
            current_entry_terms = get_entry_terms(current_entry_terms)[0]
            two_entry_sets.append(current_entry_terms)
            new_term.append(updated_term)
            which_term = two_terms.index(trm)
            count += 1
    if count == 2:
        which_term = 2
    return two_entry_sets, new_term, which_term


def create_dataset(two_entry_sets, entries_per_class):
    dat_down = dataDownload(entries_per_class=entries_per_class)
    dat_down.createdata(two_entry_sets)
    dat_down.df.to_csv("final_df.csv", index=False)
    return "Done"


if __name__ == "__main__":
    pass
    # two_entry_sets, _, _ = create_entry_terms(['adverse drug events', 'abnormalities, congenital'])
    # two_entry_sets, _, _ = create_entry_terms(['chemical reactions', 'abnormalities, congenital'])
    # print(two_entry_sets)
    # answer = create_dataset(two_entry_sets, 50)
