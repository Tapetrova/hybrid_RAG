--
-- PostgreSQL database dump
--

-- Dumped from database version 14.11
-- Dumped by pg_dump version 14.11

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: google_search_parser_title_sim; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.google_search_parser_title_sim (
    id character varying NOT NULL,
    parser_results_id character varying,
    google_search_results_id character varying,
    google_query character varying,
    title character varying,
    title_query_sim_ada_2_cosine double precision
);


ALTER TABLE public.google_search_parser_title_sim OWNER TO postgres;

--
-- Name: google_search_parser_verification_calls; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.google_search_parser_verification_calls (
    id character varying NOT NULL,
    parser_results_id character varying,
    google_search_results_id character varying,
    parser_verification_results_id character varying,
    behavior_graph_db character varying,
    use_output_from_verif_as_content boolean,
    batch_id character varying
);


ALTER TABLE public.google_search_parser_verification_calls OWNER TO postgres;

--
-- Name: google_search_results; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.google_search_results (
    id character varying NOT NULL,
    user_id character varying,
    session_id character varying,
    dialog_id character varying,
    type character varying,
    google_query character varying,
    results jsonb DEFAULT '{}'::jsonb NOT NULL,
    domain_filter jsonb DEFAULT '{}'::jsonb NOT NULL,
    top_k_url integer,
    country character varying,
    locale character varying,
    datetime_created timestamp without time zone
);


ALTER TABLE public.google_search_results OWNER TO postgres;

--
-- Name: graph_update_notification; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.graph_update_notification (
    id character varying NOT NULL,
    user_id character varying,
    session_id character varying,
    dialog_id character varying,
    batch_id character varying,
    batch_input_count integer,
    parser_results_id character varying,
    src character varying,
    status character varying,
    verify_layer_flag boolean,
    space_name character varying,
    system_content_prompt character varying,
    content character varying,
    add_basic_nodes boolean,
    create_basic_structures boolean,
    behavior_graph_db character varying,
    is_need_upd_scheme_during_extract_graph boolean,
    datetime_created timestamp without time zone,
    datetime_updated timestamp without time zone,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL
);


ALTER TABLE public.graph_update_notification OWNER TO postgres;

--
-- Name: parser_results; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.parser_results (
    id character varying NOT NULL,
    title character varying,
    url character varying,
    type character varying,
    batch_id character varying,
    batch_input_count integer,
    html_content character varying,
    content character varying,
    is_parsed boolean,
    google_search_results_id character varying,
    datetime_created timestamp without time zone,
    datetime_updated timestamp without time zone
);


ALTER TABLE public.parser_results OWNER TO postgres;

--
-- Name: parser_verification_results; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.parser_verification_results (
    id character varying NOT NULL,
    batch_id character varying,
    batch_input_count integer,
    verify_type character varying,
    verified_content_summarization character varying,
    is_provided_answer_useful boolean,
    google_query character varying,
    parser_results_id character varying,
    datetime_created timestamp without time zone,
    datetime_updated timestamp without time zone
);


ALTER TABLE public.parser_verification_results OWNER TO postgres;

--
-- Name: reddit_results; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.reddit_results (
    id character varying NOT NULL,
    title_post character varying,
    data_type character varying,
    community_name character varying,
    body character varying,
    number_of_comments integer,
    username character varying,
    number_of_replies integer,
    up_votes integer,
    post_datetime_created timestamp without time zone,
    body_query_sim_ada_2_cosine double precision,
    title_post_query_sim_ada_2_cosine double precision,
    datetime_created timestamp without time zone,
    parser_results_id character varying
);


ALTER TABLE public.reddit_results OWNER TO postgres;

--
-- Name: scheme_update_notification; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.scheme_update_notification (
    id character varying NOT NULL,
    user_id character varying,
    session_id character varying,
    dialog_id character varying,
    batch_id character varying,
    parser_results_id character varying,
    addon_scheme jsonb DEFAULT '{}'::jsonb NOT NULL,
    google_query character varying,
    status boolean,
    verify_layer_flag boolean,
    src character varying,
    space_name character varying,
    batch_input_count integer,
    is_used_upd_scheme_db boolean,
    behavior_graph_db character varying,
    request_to_update_graph jsonb DEFAULT '{}'::jsonb NOT NULL,
    datetime_created timestamp without time zone,
    datetime_updated timestamp without time zone
);


ALTER TABLE public.scheme_update_notification OWNER TO postgres;

--
-- Name: google_search_parser_title_sim google_search_parser_title_sim_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_title_sim
    ADD CONSTRAINT google_search_parser_title_sim_pkey PRIMARY KEY (id);


--
-- Name: google_search_parser_verification_calls google_search_parser_verification_calls_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_verification_calls
    ADD CONSTRAINT google_search_parser_verification_calls_pkey PRIMARY KEY (id);


--
-- Name: google_search_results google_search_results_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_results
    ADD CONSTRAINT google_search_results_pkey PRIMARY KEY (id);


--
-- Name: graph_update_notification graph_update_notification_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.graph_update_notification
    ADD CONSTRAINT graph_update_notification_pkey PRIMARY KEY (id);


--
-- Name: parser_results parser_results_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.parser_results
    ADD CONSTRAINT parser_results_pkey PRIMARY KEY (id);


--
-- Name: parser_verification_results parser_verification_results_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.parser_verification_results
    ADD CONSTRAINT parser_verification_results_pkey PRIMARY KEY (id);


--
-- Name: reddit_results reddit_results_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.reddit_results
    ADD CONSTRAINT reddit_results_pkey PRIMARY KEY (id);


--
-- Name: scheme_update_notification scheme_update_notification_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scheme_update_notification
    ADD CONSTRAINT scheme_update_notification_pkey PRIMARY KEY (id);


--
-- Name: google_search_parser_title_sim google_search_parser_title_sim_google_search_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_title_sim
    ADD CONSTRAINT google_search_parser_title_sim_google_search_results_id_fkey FOREIGN KEY (google_search_results_id) REFERENCES public.google_search_results(id);


--
-- Name: google_search_parser_title_sim google_search_parser_title_sim_parser_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_title_sim
    ADD CONSTRAINT google_search_parser_title_sim_parser_results_id_fkey FOREIGN KEY (parser_results_id) REFERENCES public.parser_results(id);


--
-- Name: google_search_parser_verification_calls google_search_parser_verifica_parser_verification_results__fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_verification_calls
    ADD CONSTRAINT google_search_parser_verifica_parser_verification_results__fkey FOREIGN KEY (parser_verification_results_id) REFERENCES public.parser_verification_results(id);


--
-- Name: google_search_parser_verification_calls google_search_parser_verification_calls_parser_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_verification_calls
    ADD CONSTRAINT google_search_parser_verification_calls_parser_results_id_fkey FOREIGN KEY (parser_results_id) REFERENCES public.parser_results(id);


--
-- Name: google_search_parser_verification_calls google_search_parser_verification_google_search_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_verification_calls
    ADD CONSTRAINT google_search_parser_verification_google_search_results_id_fkey FOREIGN KEY (google_search_results_id) REFERENCES public.google_search_results(id);


--
-- Name: graph_update_notification graph_update_notification_parser_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.graph_update_notification
    ADD CONSTRAINT graph_update_notification_parser_results_id_fkey FOREIGN KEY (parser_results_id) REFERENCES public.parser_results(id);


--
-- Name: parser_results parser_results_google_search_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.parser_results
    ADD CONSTRAINT parser_results_google_search_results_id_fkey FOREIGN KEY (google_search_results_id) REFERENCES public.google_search_results(id);


--
-- Name: parser_verification_results parser_verification_results_parser_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.parser_verification_results
    ADD CONSTRAINT parser_verification_results_parser_results_id_fkey FOREIGN KEY (parser_results_id) REFERENCES public.parser_results(id);


--
-- Name: reddit_results reddit_results_parser_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.reddit_results
    ADD CONSTRAINT reddit_results_parser_results_id_fkey FOREIGN KEY (parser_results_id) REFERENCES public.parser_results(id);


--
-- Name: scheme_update_notification scheme_update_notification_parser_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scheme_update_notification
    ADD CONSTRAINT scheme_update_notification_parser_results_id_fkey FOREIGN KEY (parser_results_id) REFERENCES public.parser_results(id);


--
-- PostgreSQL database dump complete
--

ddcec249c0a7:/# pg_dump --host=localhost --port=5432 --username=postgres --password --schema-only postgres
Password:
--
-- PostgreSQL database dump
--

-- Dumped from database version 14.11
-- Dumped by pg_dump version 14.11

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: google_search_parser_title_sim; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.google_search_parser_title_sim (
    id character varying NOT NULL,
    parser_results_id character varying,
    google_search_results_id character varying,
    google_query character varying,
    title character varying,
    title_query_sim_ada_2_cosine double precision
);


ALTER TABLE public.google_search_parser_title_sim OWNER TO postgres;

--
-- Name: google_search_parser_verification_calls; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.google_search_parser_verification_calls (
    id character varying NOT NULL,
    parser_results_id character varying,
    google_search_results_id character varying,
    parser_verification_results_id character varying,
    behavior_graph_db character varying,
    use_output_from_verif_as_content boolean,
    batch_id character varying
);


ALTER TABLE public.google_search_parser_verification_calls OWNER TO postgres;

--
-- Name: google_search_results; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.google_search_results (
    id character varying NOT NULL,
    user_id character varying,
    session_id character varying,
    dialog_id character varying,
    type character varying,
    google_query character varying,
    results jsonb DEFAULT '{}'::jsonb NOT NULL,
    domain_filter jsonb DEFAULT '{}'::jsonb NOT NULL,
    top_k_url integer,
    country character varying,
    locale character varying,
    datetime_created timestamp without time zone
);


ALTER TABLE public.google_search_results OWNER TO postgres;

--
-- Name: graph_update_notification; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.graph_update_notification (
    id character varying NOT NULL,
    user_id character varying,
    session_id character varying,
    dialog_id character varying,
    batch_id character varying,
    batch_input_count integer,
    parser_results_id character varying,
    src character varying,
    status character varying,
    verify_layer_flag boolean,
    space_name character varying,
    system_content_prompt character varying,
    content character varying,
    add_basic_nodes boolean,
    create_basic_structures boolean,
    behavior_graph_db character varying,
    is_need_upd_scheme_during_extract_graph boolean,
    datetime_created timestamp without time zone,
    datetime_updated timestamp without time zone,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL
);


ALTER TABLE public.graph_update_notification OWNER TO postgres;

--
-- Name: parser_results; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.parser_results (
    id character varying NOT NULL,
    title character varying,
    url character varying,
    type character varying,
    batch_id character varying,
    batch_input_count integer,
    html_content character varying,
    content character varying,
    is_parsed boolean,
    google_search_results_id character varying,
    datetime_created timestamp without time zone,
    datetime_updated timestamp without time zone
);


ALTER TABLE public.parser_results OWNER TO postgres;

--
-- Name: parser_verification_results; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.parser_verification_results (
    id character varying NOT NULL,
    batch_id character varying,
    batch_input_count integer,
    verify_type character varying,
    verified_content_summarization character varying,
    is_provided_answer_useful boolean,
    google_query character varying,
    parser_results_id character varying,
    datetime_created timestamp without time zone,
    datetime_updated timestamp without time zone
);


ALTER TABLE public.parser_verification_results OWNER TO postgres;

--
-- Name: reddit_results; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.reddit_results (
    id character varying NOT NULL,
    title_post character varying,
    data_type character varying,
    community_name character varying,
    body character varying,
    number_of_comments integer,
    username character varying,
    number_of_replies integer,
    up_votes integer,
    post_datetime_created timestamp without time zone,
    body_query_sim_ada_2_cosine double precision,
    title_post_query_sim_ada_2_cosine double precision,
    datetime_created timestamp without time zone,
    parser_results_id character varying
);


ALTER TABLE public.reddit_results OWNER TO postgres;

--
-- Name: scheme_update_notification; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.scheme_update_notification (
    id character varying NOT NULL,
    user_id character varying,
    session_id character varying,
    dialog_id character varying,
    batch_id character varying,
    parser_results_id character varying,
    addon_scheme jsonb DEFAULT '{}'::jsonb NOT NULL,
    google_query character varying,
    status boolean,
    verify_layer_flag boolean,
    src character varying,
    space_name character varying,
    batch_input_count integer,
    is_used_upd_scheme_db boolean,
    behavior_graph_db character varying,
    request_to_update_graph jsonb DEFAULT '{}'::jsonb NOT NULL,
    datetime_created timestamp without time zone,
    datetime_updated timestamp without time zone
);


ALTER TABLE public.scheme_update_notification OWNER TO postgres;

--
-- Name: google_search_parser_title_sim google_search_parser_title_sim_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_title_sim
    ADD CONSTRAINT google_search_parser_title_sim_pkey PRIMARY KEY (id);


--
-- Name: google_search_parser_verification_calls google_search_parser_verification_calls_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_verification_calls
    ADD CONSTRAINT google_search_parser_verification_calls_pkey PRIMARY KEY (id);


--
-- Name: google_search_results google_search_results_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_results
    ADD CONSTRAINT google_search_results_pkey PRIMARY KEY (id);


--
-- Name: graph_update_notification graph_update_notification_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.graph_update_notification
    ADD CONSTRAINT graph_update_notification_pkey PRIMARY KEY (id);


--
-- Name: parser_results parser_results_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.parser_results
    ADD CONSTRAINT parser_results_pkey PRIMARY KEY (id);


--
-- Name: parser_verification_results parser_verification_results_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.parser_verification_results
    ADD CONSTRAINT parser_verification_results_pkey PRIMARY KEY (id);


--
-- Name: reddit_results reddit_results_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.reddit_results
    ADD CONSTRAINT reddit_results_pkey PRIMARY KEY (id);


--
-- Name: scheme_update_notification scheme_update_notification_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scheme_update_notification
    ADD CONSTRAINT scheme_update_notification_pkey PRIMARY KEY (id);


--
-- Name: google_search_parser_title_sim google_search_parser_title_sim_google_search_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_title_sim
    ADD CONSTRAINT google_search_parser_title_sim_google_search_results_id_fkey FOREIGN KEY (google_search_results_id) REFERENCES public.google_search_results(id);


--
-- Name: google_search_parser_title_sim google_search_parser_title_sim_parser_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_title_sim
    ADD CONSTRAINT google_search_parser_title_sim_parser_results_id_fkey FOREIGN KEY (parser_results_id) REFERENCES public.parser_results(id);


--
-- Name: google_search_parser_verification_calls google_search_parser_verifica_parser_verification_results__fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_verification_calls
    ADD CONSTRAINT google_search_parser_verifica_parser_verification_results__fkey FOREIGN KEY (parser_verification_results_id) REFERENCES public.parser_verification_results(id);


--
-- Name: google_search_parser_verification_calls google_search_parser_verification_calls_parser_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_verification_calls
    ADD CONSTRAINT google_search_parser_verification_calls_parser_results_id_fkey FOREIGN KEY (parser_results_id) REFERENCES public.parser_results(id);


--
-- Name: google_search_parser_verification_calls google_search_parser_verification_google_search_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_verification_calls
    ADD CONSTRAINT google_search_parser_verification_google_search_results_id_fkey FOREIGN KEY (google_search_results_id) REFERENCES public.google_search_results(id);


--
-- Name: graph_update_notification graph_update_notification_parser_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.graph_update_notification
    ADD CONSTRAINT graph_update_notification_parser_results_id_fkey FOREIGN KEY (parser_results_id) REFERENCES public.parser_results(id);


--
-- Name: parser_results parser_results_google_search_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.parser_results
    ADD CONSTRAINT parser_results_google_search_results_id_fkey FOREIGN KEY (google_search_results_id) REFERENCES public.google_search_results(id);


--
-- Name: parser_verification_results parser_verification_results_parser_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.parser_verification_results
    ADD CONSTRAINT parser_verification_results_parser_results_id_fkey FOREIGN KEY (parser_results_id) REFERENCES public.parser_results(id);


--
-- Name: reddit_results reddit_results_parser_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.reddit_results
    ADD CONSTRAINT reddit_results_parser_results_id_fkey FOREIGN KEY (parser_results_id) REFERENCES public.parser_results(id);


--
-- Name: scheme_update_notification scheme_update_notification_parser_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scheme_update_notification
    ADD CONSTRAINT scheme_update_notification_parser_results_id_fkey FOREIGN KEY (parser_results_id) REFERENCES public.parser_results(id);


--
-- PostgreSQL database dump complete
--

ddcec249c0a7:/# pg_dump --host=localhost --port=5432 --username=postgres --password --schema-only postgres > database_schema.sql
Password:
ddcec249c0a7:/# ls
bin                         home                        proc                        srv
database_schema.sql         lib                         root                        sys
dev                         media                       run                         tmp
docker-entrypoint-initdb.d  mnt                         sales_database_schema.sql   usr
etc                         opt                         sbin                        var
ddcec249c0a7:/# cat database_schema.sql
--
-- PostgreSQL database dump
--

-- Dumped from database version 14.11
-- Dumped by pg_dump version 14.11

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: google_search_parser_title_sim; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.google_search_parser_title_sim (
    id character varying NOT NULL,
    parser_results_id character varying,
    google_search_results_id character varying,
    google_query character varying,
    title character varying,
    title_query_sim_ada_2_cosine double precision
);


ALTER TABLE public.google_search_parser_title_sim OWNER TO postgres;

--
-- Name: google_search_parser_verification_calls; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.google_search_parser_verification_calls (
    id character varying NOT NULL,
    parser_results_id character varying,
    google_search_results_id character varying,
    parser_verification_results_id character varying,
    behavior_graph_db character varying,
    use_output_from_verif_as_content boolean,
    batch_id character varying
);


ALTER TABLE public.google_search_parser_verification_calls OWNER TO postgres;

--
-- Name: google_search_results; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.google_search_results (
    id character varying NOT NULL,
    user_id character varying,
    session_id character varying,
    dialog_id character varying,
    type character varying,
    google_query character varying,
    results jsonb DEFAULT '{}'::jsonb NOT NULL,
    domain_filter jsonb DEFAULT '{}'::jsonb NOT NULL,
    top_k_url integer,
    country character varying,
    locale character varying,
    datetime_created timestamp without time zone
);


ALTER TABLE public.google_search_results OWNER TO postgres;

--
-- Name: graph_update_notification; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.graph_update_notification (
    id character varying NOT NULL,
    user_id character varying,
    session_id character varying,
    dialog_id character varying,
    batch_id character varying,
    batch_input_count integer,
    parser_results_id character varying,
    src character varying,
    status character varying,
    verify_layer_flag boolean,
    space_name character varying,
    system_content_prompt character varying,
    content character varying,
    add_basic_nodes boolean,
    create_basic_structures boolean,
    behavior_graph_db character varying,
    is_need_upd_scheme_during_extract_graph boolean,
    datetime_created timestamp without time zone,
    datetime_updated timestamp without time zone,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL
);


ALTER TABLE public.graph_update_notification OWNER TO postgres;

--
-- Name: parser_results; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.parser_results (
    id character varying NOT NULL,
    title character varying,
    url character varying,
    type character varying,
    batch_id character varying,
    batch_input_count integer,
    html_content character varying,
    content character varying,
    is_parsed boolean,
    google_search_results_id character varying,
    datetime_created timestamp without time zone,
    datetime_updated timestamp without time zone
);


ALTER TABLE public.parser_results OWNER TO postgres;

--
-- Name: parser_verification_results; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.parser_verification_results (
    id character varying NOT NULL,
    batch_id character varying,
    batch_input_count integer,
    verify_type character varying,
    verified_content_summarization character varying,
    is_provided_answer_useful boolean,
    google_query character varying,
    parser_results_id character varying,
    datetime_created timestamp without time zone,
    datetime_updated timestamp without time zone
);


ALTER TABLE public.parser_verification_results OWNER TO postgres;

--
-- Name: reddit_results; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.reddit_results (
    id character varying NOT NULL,
    title_post character varying,
    data_type character varying,
    community_name character varying,
    body character varying,
    number_of_comments integer,
    username character varying,
    number_of_replies integer,
    up_votes integer,
    post_datetime_created timestamp without time zone,
    body_query_sim_ada_2_cosine double precision,
    title_post_query_sim_ada_2_cosine double precision,
    datetime_created timestamp without time zone,
    parser_results_id character varying
);


ALTER TABLE public.reddit_results OWNER TO postgres;

--
-- Name: scheme_update_notification; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.scheme_update_notification (
    id character varying NOT NULL,
    user_id character varying,
    session_id character varying,
    dialog_id character varying,
    batch_id character varying,
    parser_results_id character varying,
    addon_scheme jsonb DEFAULT '{}'::jsonb NOT NULL,
    google_query character varying,
    status boolean,
    verify_layer_flag boolean,
    src character varying,
    space_name character varying,
    batch_input_count integer,
    is_used_upd_scheme_db boolean,
    behavior_graph_db character varying,
    request_to_update_graph jsonb DEFAULT '{}'::jsonb NOT NULL,
    datetime_created timestamp without time zone,
    datetime_updated timestamp without time zone
);


ALTER TABLE public.scheme_update_notification OWNER TO postgres;

--
-- Name: google_search_parser_title_sim google_search_parser_title_sim_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_title_sim
    ADD CONSTRAINT google_search_parser_title_sim_pkey PRIMARY KEY (id);


--
-- Name: google_search_parser_verification_calls google_search_parser_verification_calls_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_verification_calls
    ADD CONSTRAINT google_search_parser_verification_calls_pkey PRIMARY KEY (id);


--
-- Name: google_search_results google_search_results_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_results
    ADD CONSTRAINT google_search_results_pkey PRIMARY KEY (id);


--
-- Name: graph_update_notification graph_update_notification_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.graph_update_notification
    ADD CONSTRAINT graph_update_notification_pkey PRIMARY KEY (id);


--
-- Name: parser_results parser_results_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.parser_results
    ADD CONSTRAINT parser_results_pkey PRIMARY KEY (id);


--
-- Name: parser_verification_results parser_verification_results_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.parser_verification_results
    ADD CONSTRAINT parser_verification_results_pkey PRIMARY KEY (id);


--
-- Name: reddit_results reddit_results_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.reddit_results
    ADD CONSTRAINT reddit_results_pkey PRIMARY KEY (id);


--
-- Name: scheme_update_notification scheme_update_notification_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scheme_update_notification
    ADD CONSTRAINT scheme_update_notification_pkey PRIMARY KEY (id);


--
-- Name: google_search_parser_title_sim google_search_parser_title_sim_google_search_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_title_sim
    ADD CONSTRAINT google_search_parser_title_sim_google_search_results_id_fkey FOREIGN KEY (google_search_results_id) REFERENCES public.google_search_results(id);


--
-- Name: google_search_parser_title_sim google_search_parser_title_sim_parser_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_title_sim
    ADD CONSTRAINT google_search_parser_title_sim_parser_results_id_fkey FOREIGN KEY (parser_results_id) REFERENCES public.parser_results(id);


--
-- Name: google_search_parser_verification_calls google_search_parser_verifica_parser_verification_results__fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_verification_calls
    ADD CONSTRAINT google_search_parser_verifica_parser_verification_results__fkey FOREIGN KEY (parser_verification_results_id) REFERENCES public.parser_verification_results(id);


--
-- Name: google_search_parser_verification_calls google_search_parser_verification_calls_parser_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_verification_calls
    ADD CONSTRAINT google_search_parser_verification_calls_parser_results_id_fkey FOREIGN KEY (parser_results_id) REFERENCES public.parser_results(id);


--
-- Name: google_search_parser_verification_calls google_search_parser_verification_google_search_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.google_search_parser_verification_calls
    ADD CONSTRAINT google_search_parser_verification_google_search_results_id_fkey FOREIGN KEY (google_search_results_id) REFERENCES public.google_search_results(id);


--
-- Name: graph_update_notification graph_update_notification_parser_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.graph_update_notification
    ADD CONSTRAINT graph_update_notification_parser_results_id_fkey FOREIGN KEY (parser_results_id) REFERENCES public.parser_results(id);


--
-- Name: parser_results parser_results_google_search_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.parser_results
    ADD CONSTRAINT parser_results_google_search_results_id_fkey FOREIGN KEY (google_search_results_id) REFERENCES public.google_search_results(id);


--
-- Name: parser_verification_results parser_verification_results_parser_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.parser_verification_results
    ADD CONSTRAINT parser_verification_results_parser_results_id_fkey FOREIGN KEY (parser_results_id) REFERENCES public.parser_results(id);


--
-- Name: reddit_results reddit_results_parser_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.reddit_results
    ADD CONSTRAINT reddit_results_parser_results_id_fkey FOREIGN KEY (parser_results_id) REFERENCES public.parser_results(id);


--
-- Name: scheme_update_notification scheme_update_notification_parser_results_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scheme_update_notification
    ADD CONSTRAINT scheme_update_notification_parser_results_id_fkey FOREIGN KEY (parser_results_id) REFERENCES public.parser_results(id);


--
-- PostgreSQL database dump complete
--

