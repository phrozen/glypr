/// Official Task Prompts for EmbeddingGemma 2
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Task<'a> {
    #[default]
    SearchQuery,
    Document {
        title: Option<&'a str>,
    },
    CodeRetrieval,
    Classification,
    Clustering,
    QuestionAnswering,
    FactChecking,
    SentenceSimilarity,
    Custom(&'a str),
    Raw,
}

impl<'a> Task<'a> {
    /// Returns the official prompt prefix string for this task, if any.
    pub fn prompt_prefix(&self) -> Option<String> {
        match self {
            Task::SearchQuery => Some("task: search result | query: ".to_string()),
            Task::Document { title } => match title {
                Some(t) => Some(format!("title: {} | text: ", t)),
                None => Some("title: none | text: ".to_string()),
            },
            Task::CodeRetrieval => Some("task: code retrieval | query: ".to_string()),
            Task::Classification => Some("task: classification | query: ".to_string()),
            Task::Clustering => Some("task: clustering | query: ".to_string()),
            Task::QuestionAnswering => Some("task: question answering | query: ".to_string()),
            Task::FactChecking => Some("task: fact checking | query: ".to_string()),
            Task::SentenceSimilarity => Some("task: sentence similarity | query: ".to_string()),
            Task::Custom(s) => {
                if s.contains(':') || s.contains('|') {
                    Some(s.to_string())
                } else {
                    Some(format!("task: {} | query: ", s))
                }
            }
            Task::Raw => None,
        }
    }

    /// Format a text string with this task's prompt prefix.
    pub fn format_text(&self, text: &str) -> String {
        match self.prompt_prefix() {
            Some(prefix) => format!("{}{}", prefix, text),
            None => text.to_string(),
        }
    }

    /// Construct a Task from a name string (case-insensitive) and an optional document title.
    pub fn from_name_and_title(name: &'a str, title: Option<&'a str>) -> Self {
        match name.to_lowercase().replace(['-', '_', ' '], "").as_str() {
            "searchquery" | "retrieval" | "retrievalquery" | "query" | "reranking"
            | "bitextmining" => Task::SearchQuery,
            "document" | "retrievaldocument" => Task::Document { title },
            "coderetrieval" | "code" => Task::CodeRetrieval,
            "classification" | "multilabelclassification" => Task::Classification,
            "clustering" => Task::Clustering,
            "questionanswering" | "qa" => Task::QuestionAnswering,
            "factchecking" => Task::FactChecking,
            "sentencesimilarity" | "sts" | "pairclassification" | "summarization" => {
                Task::SentenceSimilarity
            }
            "raw" | "none" => Task::Raw,
            _ => Task::Custom(name),
        }
    }

    /// Convert this borrowed `Task<'a>` into an owned `OwnedTask`.
    pub fn to_owned(&self) -> OwnedTask {
        match self {
            Task::SearchQuery => OwnedTask::SearchQuery,
            Task::Document { title } => OwnedTask::Document {
                title: title.map(|s| s.to_string()),
            },
            Task::CodeRetrieval => OwnedTask::CodeRetrieval,
            Task::Classification => OwnedTask::Classification,
            Task::Clustering => OwnedTask::Clustering,
            Task::QuestionAnswering => OwnedTask::QuestionAnswering,
            Task::FactChecking => OwnedTask::FactChecking,
            Task::SentenceSimilarity => OwnedTask::SentenceSimilarity,
            Task::Custom(s) => OwnedTask::Custom((*s).to_string()),
            Task::Raw => OwnedTask::Raw,
        }
    }

    /// Canonical task names and descriptions/aliases for CLI help and introspection.
    pub const fn all_names() -> &'static [(&'static str, &'static str)] {
        &[
            (
                "search_query",
                "Search / RAG query (aliases: retrieval, retrieval_query, query, reranking, bitextmining)",
            ),
            (
                "document",
                "Document / passage with optional title (aliases: retrieval_document)",
            ),
            ("code_retrieval", "Code search query (aliases: code)"),
            (
                "question_answering",
                "Question answering query (aliases: qa)",
            ),
            ("fact_checking", "Fact checking claim"),
            (
                "classification",
                "Text classification (aliases: multilabel_classification)",
            ),
            ("clustering", "Topic clustering"),
            (
                "sentence_similarity",
                "Semantic similarity / STS (aliases: sts, pair_classification, summarization)",
            ),
            ("raw", "Unprompted raw embedding (aliases: none)"),
        ]
    }
}

/// Owned representation of `Task` that can move across thread boundaries without lifetimes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum OwnedTask {
    #[default]
    SearchQuery,
    Document {
        title: Option<String>,
    },
    CodeRetrieval,
    Classification,
    Clustering,
    QuestionAnswering,
    FactChecking,
    SentenceSimilarity,
    Custom(String),
    Raw,
}

impl OwnedTask {
    /// Borrow as a `Task<'_>`.
    pub fn as_task(&self) -> Task<'_> {
        match self {
            Self::SearchQuery => Task::SearchQuery,
            Self::Document { title } => Task::Document {
                title: title.as_deref(),
            },
            Self::CodeRetrieval => Task::CodeRetrieval,
            Self::Classification => Task::Classification,
            Self::Clustering => Task::Clustering,
            Self::QuestionAnswering => Task::QuestionAnswering,
            Self::FactChecking => Task::FactChecking,
            Self::SentenceSimilarity => Task::SentenceSimilarity,
            Self::Custom(s) => Task::Custom(s),
            Self::Raw => Task::Raw,
        }
    }

    /// Construct an `OwnedTask` from a name string (case-insensitive) and an optional document title.
    pub fn from_name_and_title(name: &str, title: Option<String>) -> Self {
        match Task::from_name_and_title(name, title.as_deref()) {
            Task::SearchQuery => Self::SearchQuery,
            Task::Document { .. } => Self::Document { title },
            Task::CodeRetrieval => Self::CodeRetrieval,
            Task::Classification => Self::Classification,
            Task::Clustering => Self::Clustering,
            Task::QuestionAnswering => Self::QuestionAnswering,
            Task::FactChecking => Self::FactChecking,
            Task::SentenceSimilarity => Self::SentenceSimilarity,
            Task::Custom(s) => Self::Custom(s.to_string()),
            Task::Raw => Self::Raw,
        }
    }

    /// Format a text string with this task's prompt prefix.
    pub fn format_text(&self, text: &str) -> String {
        self.as_task().format_text(text)
    }

    /// Returns the official prompt prefix string for this task, if any.
    pub fn prompt_prefix(&self) -> Option<String> {
        self.as_task().prompt_prefix()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_prompt_formatting() {
        assert_eq!(
            Task::SearchQuery.format_text("quantum physics"),
            "task: search result | query: quantum physics"
        );
        assert_eq!(
            Task::Document { title: None }.format_text("content body"),
            "title: none | text: content body"
        );
        assert_eq!(
            Task::Document {
                title: Some("Physics 101")
            }
            .format_text("content body"),
            "title: Physics 101 | text: content body"
        );
        assert_eq!(
            Task::CodeRetrieval.format_text("fn test()"),
            "task: code retrieval | query: fn test()"
        );
        assert_eq!(
            Task::Classification.format_text("sentiment review"),
            "task: classification | query: sentiment review"
        );
        assert_eq!(Task::Raw.format_text("unprompted text"), "unprompted text");
    }

    #[test]
    fn test_task_from_name_and_title() {
        assert_eq!(
            Task::from_name_and_title("SearchQuery", None),
            Task::SearchQuery
        );
        assert_eq!(
            Task::from_name_and_title("document", Some("Doc A")),
            Task::Document {
                title: Some("Doc A")
            }
        );
        assert_eq!(
            Task::from_name_and_title("code_retrieval", None),
            Task::CodeRetrieval
        );
        assert_eq!(
            Task::from_name_and_title("qa", None),
            Task::QuestionAnswering
        );
        assert_eq!(
            Task::from_name_and_title("reranking", None),
            Task::SearchQuery
        );
        assert_eq!(
            Task::from_name_and_title("bitextmining", None),
            Task::SearchQuery
        );
        assert_eq!(Task::from_name_and_title("raw", None), Task::Raw);
        assert_eq!(Task::from_name_and_title("none", None), Task::Raw);

        let owned = OwnedTask::from_name_and_title("document", Some("My Title".to_string()));
        assert_eq!(
            owned.as_task(),
            Task::Document {
                title: Some("My Title")
            }
        );
        assert_eq!(owned.as_task().to_owned(), owned);
        assert!(!Task::all_names().is_empty());
    }
}
