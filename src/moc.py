def auto_generate_MOCs(vault_nlp, *, out_dir, top_n_docs):
    n_topics = vault_nlp.anchor_topic_model.n_hidden
    for n in range(n_topics):
        top_docs = (vault_nlp.anchor_topic_model
                    .get_top_docs(topic=n, n_docs=top_n_docs, sort_by='tc'))
        list_links = [f"# Topic {n}\nAnchor words: {vault_nlp.anchor_words[n]}"]
        for link, _ in top_docs:  # <-- this unpacks the tuple like a, b = (0, 1)
            list_links.append("".join(["- [[", link, "]]"]))

        # export:
        file_text = "\n".join(list_links)
        with open(out_dir / f"topic_{n}.md", 'w') as f:
            f.write(file_text)
    pass
